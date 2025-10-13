"""
数据集：细胞周期数据集，面向细胞周期做一个三分类任务，即G1、S、G2/M
任务：使用细胞周期数据集微调预训练模型KCellFM，之后在测试集上测试微调模型的性能
输出：微调模型在测试集上的分类性能
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab
from sklearn.model_selection import train_test_split
import anndata
import random
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt


# 固定随机种子保证可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 配置参数
epochs = 100  # 修改为50个epoch
batch_size = 64
embsize = 512
nhead = 8
d_hid = 512
nlayers = 6
dropout = 0.1
lr = 1e-5
pad_token = "<pad>"
max_seq_len = 8192
input_emb_style = "continuous"
cell_emb_style = "cls"
mask_value = -1
pad_value = -2

# 加载词汇表
vocab = GeneVocab.from_file("/home/lxz/scmamba/vocab.json")
ntokens = len(vocab)

# 直接使用原始标签进行分类
label_to_id = {
    'single H1-Fucci cell sorted from G1 phase of the cell cycle only': 0,
    'single H1-Fucci cell sorted from S phase of the cell cycle only': 1,
    'single H1-Fucci cell sorted from G2/M phase of the cell cycle only': 2
}
class_num = len(label_to_id)


class CellCycleDataset(Dataset):
    def __init__(self, expr_matrix, labels):
        self.expr_matrix = expr_matrix
        self.labels = labels
        self.cell_ids = list(labels.index)

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        # 获取基因表达数据
        expr_values = self.expr_matrix.loc[cell_id].values  # 确保内存连续
        gene_names = self.expr_matrix.columns

        # 只保留表达值非零的基因
        non_zero_indices = np.where(expr_values != 0)[0]
        expr_values = expr_values[non_zero_indices]
        gene_names = gene_names[non_zero_indices]

        # 将基因名称映射为ID，并同时过滤表达值
        filtered_gene_ids = []
        filtered_expr_values = []
        for gene, value in zip(gene_names, expr_values):
            if gene in vocab:  # 只保留在词汇表中的基因
                filtered_gene_ids.append(vocab[gene])
                filtered_expr_values.append(value)
        gene_ids = filtered_gene_ids
        expr_values = filtered_expr_values

        # 添加CLS token
        if len(gene_ids) > max_seq_len:
            idx = np.random.choice(len(gene_ids), max_seq_len, replace=False)
            gene_ids = [gene_ids[i] for i in idx]
            expr_values = [expr_values[i] for i in idx]

        # 填充序列
        padding_length = max_seq_len - len(gene_ids)
        if padding_length > 0:
            gene_ids = gene_ids + [vocab["<pad>"]] * padding_length
            expr_values = expr_values + [pad_value] * padding_length
        gene_ids = [vocab["<cls>"]] + gene_ids
        expr_values = np.concatenate([[0.0], expr_values])

        # 创建padding mask
        padding_mask = [False] * (max_seq_len + 1)
        for i in range(len(padding_mask)):
            if gene_ids[i] == vocab["<pad>"]:
                padding_mask[i] = True

        # 获取标签
        label = label_to_id[self.labels[cell_id]]
        return {
            'gene_ids': torch.LongTensor(gene_ids),
            'expr_values': torch.FloatTensor(expr_values),
            'padding_mask': torch.BoolTensor(padding_mask),
            'label': torch.as_tensor(label, dtype=torch.long)
        }


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    expr_matrix = pd.read_csv("/home/lxz/scmamba/细胞周期/h-ESC/GSE64016_logp_10k.csv", index_col=0)
    adata = anndata.read_h5ad("/home/lxz/scmamba/细胞周期/GSE64016_adata.h5ad")
    expr_matrix.index = adata.obs.index
    labels = pd.Series(adata.obs['source_name_ch1'], index=adata.obs.index)
    mask = labels != 'single H1 hESC'
    filtered_expr_matrix = expr_matrix[mask]
    filtered_labels = labels[mask]

    # 划分训练测试集
    train_idx, test_idx = train_test_split(
        np.arange(len(filtered_labels)),
        test_size=0.3,
        stratify=filtered_labels,
        random_state=42
    )

    # 创建数据集
    full_dataset = CellCycleDataset(filtered_expr_matrix, filtered_labels)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = MambaModel(
        ntokens, embsize, nhead, d_hid, nlayers,
        vocab=vocab, dropout=dropout, pad_token=pad_token,
        pad_value=pad_value, input_emb_style=input_emb_style,
        cell_emb_style=cell_emb_style, class_num=class_num
    ).to(device)

    # 加载预训练权重（跳过分类头）
    try:
        pretrained_dict = torch.load("/home/lxz/scmamba/model_state/cell_cls_3loss_6layer_final.pth",
                                     map_location=device)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape
                           and not k.startswith('cls_decoder')}  # 排除所有分类头参数

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("成功加载预训练权重 (分类头权重除外)")

        # 重新初始化分类头
        print("初始化分类头权重...")
        nn.init.kaiming_normal_(model.cls_decoder.out_layer.weight, mode='fan_in', nonlinearity='relu')
        if model.cls_decoder.out_layer.bias is not None:
            nn.init.zeros_(model.cls_decoder.out_layer.bias)
    except Exception as e:
        print(f"加载预训练权重失败: {str(e)}")

    # 冻结和解冻逻辑
    print(f"冻结前 {nlayers - 3} 层，解冻最后3层和分类器")

    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后两层Mamba层
    for i in range(nlayers - 3, nlayers):
        layer = model.mamba_encoder[i]
        for param in layer.parameters():
            param.requires_grad = True
        print(f"解冻Mamba层 {i}: 参数可训练")

    # 解冻分类头
    for param in model.cls_decoder.parameters():
        param.requires_grad = True
    print("分类器参数已设置为可训练")

    # 打印可训练参数
    print("\n可训练参数:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # 只优化需要训练的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # 记录训练过程
    test_accuracies = []
    train_losses = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            gene_ids = batch['gene_ids'].to(device)
            expr_values = batch['expr_values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(
                    src=gene_ids,
                    values=expr_values,
                    src_key_padding_mask=padding_mask
                )
            loss = criterion(outputs["cls_output"], labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 记录训练损失
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 测试集评估
        model.eval()
        test_preds, test_labels = [], []
        all_cell_embs = []
        with torch.no_grad():
            for batch in test_loader:
                gene_ids = batch['gene_ids'].to(device)
                expr_values = batch['expr_values'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(src=gene_ids, values=expr_values, src_key_padding_mask=padding_mask)
                _, predicted = torch.max(outputs["cls_output"], 1)
                cell_embs = outputs["cell_emb"].cpu().numpy()
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                all_cell_embs.append(cell_embs)

        test_accuracy = 100 * np.sum(np.array(test_preds) == np.array(test_labels)) / len(test_labels)
        test_accuracies.append(test_accuracy)
        all_cell_embs = np.concatenate(all_cell_embs, axis=0)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

    # 保存模型和结果
    torch.save(model.state_dict(), '/home/lxz/scmamba/model_state/mamba_cell_cycle_final.pth')
    np.save("/home/lxz/scmamba/细胞周期/cell_embeddings.npy", all_cell_embs)

    predictions = {
        'preds': test_preds,
        'labels': test_labels,
    }

    with open('/home/lxz/scmamba/细胞周期/labels_preds.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), test_accuracies, 'b-o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs. Epoch')
    plt.xticks(np.arange(1, epochs + 1, step=max(1, epochs // 10)))
    plt.grid(True)
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_losses, 'r-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs. Epoch')
    plt.xticks(np.arange(1, epochs + 1, step=max(1, epochs // 10)))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('/home/lxz/scmamba/细胞周期/training_curve.png', dpi=300)
    plt.show()

    # 输出最终结果
    print("\n===== Final Test Results =====")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=list(label_to_id.keys()),
        digits=4
    ))


if __name__ == "__main__":
    train()