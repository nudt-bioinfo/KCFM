"""
数据集：空转数据集（基因数量少）
数据子类型：空转SB_intra
输出：基于空转SB_intra微调的预训练模型以及微调模型在验证集上的分类报告
"""
import sys
from pathlib import Path

# 获取项目根目录，假设当前文件在“空转”文件夹下，根目录是“空转”的上级目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.model import MambaModel  # 保持导入位置，避免循环导入
from models.gene_tokenizer import GeneVocab


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 配置参数
class Config:
    # 数据参数
    train_data_path = "/home/lxz/scmamba/空转/Hubmap_SB_intra/train.csv"
    model_save_dir = "/home/lxz/scmamba/model_state/"
    os.makedirs(model_save_dir, exist_ok=True)  # 确保保存目录存在

    # 模型参数
    embsize = 512
    nhead = 8
    d_hid = 512
    nlayers = 6
    dropout = 0.1
    pad_token = "<pad>"
    max_seq_len = 50
    input_emb_style = "continuous"
    cell_emb_style = "cls"
    mask_value = -1
    pad_value = -2
    vocab_path = "/home/lxz/scmamba/vocab.json"
    pretrained_model_path = "/home/lxz/scmamba/model_state/cell_cls_3loss_6layer_final.pth"

    # 训练参数
    epochs = 10
    batch_size = 96  # 96
    lr = 5e-5  # 5e-5 ~ 2e-4,
    weight_decay = 1e-4
    val_split = 0.3  # 验证集比例
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 类别映射
    celltype_to_id = {
        "CD4T": 0,
        "CD7_Immune": 1,
        "CD8T": 2,
        "DC": 3,
        "Endothelial": 4,
        "Goblet": 5,
        "ICC": 6,
        "Lymphatic": 7,
        "Macrophage": 8,
        "Nerve": 9,
        "Neuroendocrine": 10,
        "Neutrophil": 11,
        "Plasma": 12,
        "Stroma": 13,
        "TA": 14
    }
    class_num = len(celltype_to_id)

    # 基因映射
    gene_to_id = {
        "MUC2": 17183, "SOX9": 32052, "MUC1": 17175, "CD31": 19330, "Synapto": 32742,
        "CD49f": 12272, "CD15": 9687, "CHGA": 4894, "CDX2": 4568, "ITLN1": 12308,
        "CD4": 4380, "CD127": 12051, "Vimentin": 35192, "HLADR": 11044, "CD8": 4412,
        "CD11c": 12283, "CD44": 4383, "CD16": 9286, "BCL2": 3080, "CD123": 36627,
        "CD38": 4376, "CD90": 33320, "aSMA": 1391, "CD21": 5523, "NKG2D": 12911,
        "CD66": 4589, "CD57": 2956, "CD206": 16892, "CD68": 4397, "CD34": 4373,
        "aDef5": 7546, "CD7": 4399, "CD36": 4374, "CD138": 30796, "Cytokeratin": 41736,
        "CK7": 12989, "CD117": 12801, "CD19": 4335, "Podoplanin": 19298, "CD45": 20664,
        "CD56": 17477, "CD69": 4398, "Ki67": 16711, "CD49a": 12264, "CD163": 4329,
        "CD161": 12901
    }


config = Config()


# 数据集定义
class SpatialDataset(Dataset):
    def __init__(self, csv_file, gene_to_id, celltype_to_id):
        self.data = pd.read_csv(csv_file)
        self.gene_to_id = gene_to_id
        self.celltype_to_id = celltype_to_id
        # 筛选数据中存在的基因
        self.genes = [col for col in self.data.columns if col in gene_to_id]
        # 过滤无效细胞类型
        valid_cell_types = set(celltype_to_id.keys())
        self.data = self.data[self.data['cell_type_A'].isin(valid_cell_types)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 获取基因ID和表达值
        vocab_ids = []
        expr_values = []
        for gene in self.genes:
            vocab_ids.append(self.gene_to_id[gene])
            expr_values.append(row[gene])

        # 添加CLS token（假设60695是CLS token ID）
        vocab_ids = [60695] + vocab_ids
        expr_values = [0.0] + expr_values

        # 序列填充/截断
        if len(vocab_ids) > config.max_seq_len:
            vocab_ids = vocab_ids[:config.max_seq_len]
            expr_values = expr_values[:config.max_seq_len]
        else:
            padding_length = config.max_seq_len - len(vocab_ids)
            vocab_ids += [60694] * padding_length  # 60694是pad token ID
            expr_values += [config.pad_value] * padding_length

        # 创建padding mask（True表示需要mask）
        padding_mask = [False] * config.max_seq_len
        for i in range(len(vocab_ids)):
            if vocab_ids[i] == 60694:
                padding_mask[i] = True

        # 获取细胞类型ID
        cell_type = row['cell_type_A']
        cell_type_id = self.celltype_to_id[cell_type]

        # 空间坐标
        x, y = row['x'], row['y']

        return {
            'src': torch.tensor(vocab_ids, dtype=torch.long),
            'values': torch.tensor(expr_values, dtype=torch.float),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'celltype': torch.tensor(cell_type_id, dtype=torch.long),
            'coordinates': torch.tensor([x, y], dtype=torch.float)
        }


# 评估函数（修正键名错误，移除多余参数）
def evaluate(model, loader, device, criterion_cls):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)  # 修正键名：cell_type -> celltype

            with torch.cuda.amp.autocast(enabled=True):
                model_output = model(
                    src=src,
                    values=values,
                    src_key_padding_mask=padding_mask
                )
                loss = criterion_cls(model_output["cls_output"], cell_types)
                total_loss += loss.item() * src.size(0)  # 按批次大小加权

                preds = torch.argmax(model_output["cls_output"], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cell_types.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels


def train():
    # 设备设置
    device = config.device
    torch.cuda.set_device(device)
    print(f"使用设备: {device}")

    # 加载数据集
    full_dataset = SpatialDataset(
        csv_file=config.train_data_path,
        gene_to_id=config.gene_to_id,
        celltype_to_id=config.celltype_to_id
    )
    print(f"总训练数据样本数: {len(full_dataset)}")

    # 正确获取所有标签用于分层抽样（修正标签获取方式）
    all_labels = [full_dataset[i]['celltype'].item() for i in range(len(full_dataset))]

    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=config.val_split,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 初始化模型
    vocab = GeneVocab.from_file(config.vocab_path)
    ntokens = len(vocab)

    model = MambaModel(
        ntokens, config.embsize, config.nhead, config.d_hid, config.nlayers,
        dropout=config.dropout, pad_token=config.pad_token,
        pad_value=config.pad_value, input_emb_style=config.input_emb_style,
        cell_emb_style=config.cell_emb_style, class_num=config.class_num
    ).to(device)

    # 加载预训练权重
    try:
        pretrained_dict = torch.load(config.pretrained_model_path, map_location=device)
        model_dict = model.state_dict()

        # 过滤可加载的权重（排除分类头）
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and 'cls_decoder' not in k
        }

        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)

        print(f"成功加载 {len(pretrained_dict_filtered)} 个预训练层权重")
        print("初始化分类头权重...")
        # 重新初始化分类头
        nn.init.kaiming_normal_(model.cls_decoder.out_layer.weight, mode='fan_in', nonlinearity='relu')
        if model.cls_decoder.out_layer.bias is not None:
            nn.init.zeros_(model.cls_decoder.out_layer.bias)
    except Exception as e:
        print(f"加载预训练权重失败: {str(e)}，将使用随机初始化权重")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    criterion_cls = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 训练循环
    best_val_acc = 0.0
    best_model_path = os.path.join(config.model_save_dir, "spatial_classifier_SB_intra_best.pth")

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                model_output = model(
                    src=src,
                    values=values,
                    src_key_padding_mask=padding_mask
                )
                loss = criterion_cls(model_output["cls_output"], cell_types)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * src.size(0)
            pbar.set_postfix(train_loss=loss.item())

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataset)

        # 验证
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion_cls)

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
    _, _, val_preds, val_labels = evaluate(model, val_loader, device, criterion_cls)
    print("\n最佳模型在验证集上的详细报告:")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=config.celltype_to_id.keys(),
        digits=4,
        zero_division=0
    ))


if __name__ == "__main__":
    train()