"""
数据集：T cancer cell训练数据集
任务：使用T cancer cell数据集微调KCellFM预训练模型，实现T cancer cell classification
输出：微调模型
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import scanpy as sc
import numpy as np
from scipy import sparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 配置参数
epochs = 3
batch_size = 32
embsize = 512
nhead = 8
d_hid = 512
nlayers = 6  # 总层数
fine_tune_layers = 2  # 微调最后两层
dropout = 0.1
lr =2e-4  # 1e-5
weight_decay = 1e-3  # 1e-4
pad_token = "<pad>"
max_seq_len = 4096
input_emb_style = "continuous"
cell_emb_style = "cls"
mask_value = -1
pad_value = -2
val_split = 0.3  # 验证集比例
model_save_dir = "/home/lxz/scmamba/model_state/"
os.makedirs(model_save_dir, exist_ok=True)  # 确保保存目录存在

# 加载词汇表
vocab = GeneVocab.from_file("/home/lxz/scmamba/vocab.json")
ntokens = len(vocab)

# 细胞类型映射
celltype_to_id = {
    'T cell': 0,
    'CD8-positive, alpha-beta cytotoxic T cell': 1,
    'naive thymus-derived CD4-positive, alpha-beta T cell': 2,
    'effector CD8-positive, alpha-beta T cell': 3,
    'effector memory CD8-positive, alpha-beta T cell': 4,
    'central memory CD4-positive, alpha-beta T cell': 5,
    'gamma-delta T cell': 6
}
class_num = len(celltype_to_id)


class SingleCellDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata
        self.cell_ids = adata.obs_names.tolist()
        self.gene_names = adata.var.feature_name.tolist()

        # 预计算非零表达基因的索引（加速处理）
        self.nonzero_indices = {}
        expr_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        for i, cell_id in enumerate(self.cell_ids):
            self.nonzero_indices[cell_id] = np.where(expr_matrix[i] != 0)[0]

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_type = self.adata.obs.loc[cell_id, 'cell_type']

        # 获取非零表达的基因和值
        nonzero_idx = self.nonzero_indices[cell_id]
        expr_values = self.adata.X[idx, nonzero_idx].toarray().flatten() \
            if sparse.issparse(self.adata.X) \
            else self.adata.X[idx, nonzero_idx]
        gene_names = [self.gene_names[i] for i in nonzero_idx]

        # 映射基因名到ID，并过滤不在词汇表中的基因
        gene_ids = []
        filtered_expr = []
        for gene, value in zip(gene_names, expr_values):
            if gene in vocab:
                gene_ids.append(vocab[gene])
                filtered_expr.append(value)

        # 随机截断（如果超过最大长度）
        if len(gene_ids) > max_seq_len - 1:  # -1 为CLS预留位置
            selected = np.random.choice(len(gene_ids), max_seq_len - 1, replace=False)
            gene_ids = [gene_ids[i] for i in selected]
            filtered_expr = [filtered_expr[i] for i in selected]

        # 添加CLS标记（ID和表达值）
        gene_ids = [vocab["<cls>"]] + gene_ids
        filtered_expr = [0.0] + filtered_expr  # CLS的表达值设为0

        # 填充到固定长度
        padding_len = max_seq_len - len(gene_ids)
        if padding_len > 0:
            gene_ids += [vocab["<pad>"]] * padding_len
            filtered_expr += [pad_value] * padding_len

        # 创建padding mask（True表示填充位置）
        padding_mask = [id_ == vocab["<pad>"] for id_ in gene_ids]

        return {
            'src': torch.LongTensor(gene_ids),
            'values': torch.FloatTensor(filtered_expr),
            'padding_mask': torch.BoolTensor(padding_mask),
            'celltype': torch.tensor(celltype_to_id[cell_type], dtype=torch.long)
        }


# 评估函数
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                model_output = model(
                    src=src,
                    values=values,
                    src_key_padding_mask=padding_mask
                )
                loss = criterion(model_output["cls_output"], cell_types)
                total_loss += loss.item() * src.size(0)  # 按批次大小加权

                preds = torch.argmax(model_output["cls_output"], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cell_types.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 加载数据
    print("开始加载h5ad文件...")
    adata_train = sc.read("/mnt/HHD16T/DATA/lxz/cancer/T_train.h5ad")
    print("h5ad文件加载完成，细胞数量:", adata_train.n_obs)

    print("开始构建数据集...")
    full_dataset = SingleCellDataset(adata_train)
    print("数据集构建完成，总样本数:", len(full_dataset))

    # 获取所有标签用于分层抽样
    all_labels = [full_dataset[i]['celltype'].item() for i in range(len(full_dataset))]

    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=val_split,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        # pin_memory=True,
        # drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        # drop_last=False
    )

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
                           and not k.startswith('cls_decoder.out_layer')}

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

    # 冻结和解冻逻辑：针对mamba_encoder中的层
    print(f"冻结前 {nlayers - fine_tune_layers} 层，只微调最后 {fine_tune_layers} 层和分类器")

    # 检查mamba_encoder结构是否符合预期
    if hasattr(model, 'mamba_encoder') and isinstance(model.mamba_encoder, nn.ModuleList):
        # 验证层数是否匹配配置
        actual_layers = len(model.mamba_encoder)
        if actual_layers != nlayers:
            raise ValueError(f"模型mamba_encoder层数与配置不符，实际为{actual_layers}，预期为{nlayers}")

        # 冻结前面的层
        for i in range(nlayers - fine_tune_layers):
            layer = model.mamba_encoder[i]
            for param in layer.parameters():
                param.requires_grad = False
            print(f"冻结层 {i}: 参数已冻结")

        # 解冻最后两层
        for i in range(nlayers - fine_tune_layers, nlayers):
            layer = model.mamba_encoder[i]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"解冻层 {i}: 参数可训练")
    else:
        raise AttributeError("模型中未找到mamba_encoder或其不是nn.ModuleList类型")

    # 确保分类器可训练
    if hasattr(model, 'cls_decoder'):
        for param in model.cls_decoder.parameters():
            param.requires_grad = True
        print("分类器参数已设置为可训练")
    else:
        raise AttributeError("模型中未找到cls_decoder分类器")

    # 只优化需要训练的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    # 优化器和损失函数（只传入可训练参数）
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 训练循环
    best_val_acc = 0.0
    best_model_path = os.path.join(model_save_dir, "cancer_Tcell_2_layers_best.pth")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                output = model(src=src, values=values, src_key_padding_mask=padding_mask)
                loss = criterion(output["cls_output"], cell_types)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  # 只裁剪可训练参数的梯度
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * src.size(0)
            pbar.set_postfix({'loss': loss.item()})

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataset)

        # 验证
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion)

        # 打印 epoch 结果
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"保存新的最佳模型 (验证准确率: {best_val_acc:.4f}) 至 {best_model_path}")

    # 训练结束后打印最佳结果
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f} (模型保存于 {best_model_path})")

    # 加载最佳模型并打印验证集详细报告
    model.load_state_dict(torch.load(best_model_path))
    _, _, val_preds, val_labels = evaluate(model, val_loader, device, criterion)
    print("\n最佳模型在验证集上的详细报告:")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=celltype_to_id.keys(),
        digits=4,
        zero_division=0
    ))

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "cancer_Tcell_final_2_layers_10_epochs.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存为 {final_model_path}")


if __name__ == "__main__":
    train()