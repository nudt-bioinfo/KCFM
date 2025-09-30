"""
数据集：空转数据集的测试集，包含4种类型(CL_cross、CL_intra、SB_cross、SB_intra)
输出：微调模型在4种空转数据集的测试集上的分类性能表现，并输出真实、预测标签以及所有测试样本的细胞嵌入表示
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab
import pickle

# 配置参数 (与训练保持一致)
batch_size = 96
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

# 加载词汇表和基因映射
vocab = GeneVocab.from_file("/home/lxz/scmamba/vocab.json")
ntokens = len(vocab)

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

CL_cross ={
    "CD4T": 0,
    "CD7_Immune": 1,
    "CD8T": 2,
    "DC": 3,
    "Endothelial": 4,
    "Enterocyte_ITLN1p": 5,
    "Goblet": 6,
    "ICC": 7,
    "Lymphatic": 8,
    "Macrophage": 9,
    "Nerve": 10,
    "Neuroendocrine": 11,
    "Neutrophil": 12,
    "Paneth": 13,
    "Plasma": 14,
    "Stroma": 15,
    "TA": 16
}

CL_intra = {
    "CD4T": 0,
    "CD8T": 1,
    "DC": 2,
    "Endothelial": 3,
    "Enterocyte_ITLN1p": 4,
    "Goblet": 5,
    "ICC": 6,
    "Lymphatic": 7,
    "Macrophage": 8,
    "Nerve": 9,
    "Neuroendocrine": 10,
    "SmoothMuscle": 11,
    "Stroma": 12,
    "TA": 13
}


SB_cross = {
    "B": 0,
    "CD4T": 1,
    "CD7_Immune": 2,
    "CD8T": 3,
    "DC": 4,
    "Endothelial": 5,
    "Enterocyte_ITLN1p": 6,
    "Goblet": 7,
    "ICC": 8,
    "Lymphatic": 9,
    "Macrophage": 10,
    "Nerve": 11,
    "Neuroendocrine": 12,
    "Neutrophil": 13,
    "Plasma": 14,
    "Stroma": 15,
    "TA": 16
}

SB_intra = {
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

# 细胞类型映射 (必须与训练时完全一致)
# 空转共有4种类型的数据，包括CL_cross、CL_intra、SB_cross、SB_intra，在此处修改需要测试的数据类型
celltype_to_id = CL_cross

id_to_celltype = {v: k for k, v in celltype_to_id.items()}
class_num = len(celltype_to_id)

# 测试数据集类 (与训练相同)
class TestDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.genes = [col for col in self.data.columns if col in gene_to_id]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        vocab_ids = []
        expr_values = []
        for gene in self.genes:
            vocab_ids.append(gene_to_id[gene])
            expr_values.append(row[gene])
        
        vocab_ids = [60695] + vocab_ids  # <cls> token
        expr_values = [0.0] + expr_values
        
        padding_length = max_seq_len - len(vocab_ids)
        if padding_length > 0:
            vocab_ids = vocab_ids + [60694] * padding_length  # <pad> token
            expr_values = expr_values + [pad_value] * padding_length
        
        padding_mask = [False] * len(vocab_ids)
        for i in range(len(vocab_ids)):
            if vocab_ids[i] == 60694:
                padding_mask[i] = True
        
        cell_type = row['cell_type_A']
        cell_type_id = celltype_to_id.get(cell_type, -1)  # -1表示未知类型
        
        return {
            'src': torch.tensor(vocab_ids, dtype=torch.long),
            'values': torch.tensor(expr_values, dtype=torch.float),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'celltype': torch.tensor(cell_type_id, dtype=torch.long),
            'coordinates': torch.tensor([row['x'], row['y']], dtype=torch.float)
        }

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = MambaModel(
        ntokens, embsize, nhead, d_hid, nlayers,
        dropout=dropout, pad_token=pad_token,
        pad_value=pad_value, input_emb_style=input_emb_style,
        cell_emb_style=cell_emb_style, class_num=class_num
    ).to(device)
    
    # 加载训练好的模型权重
    model_path = "/home/lxz/scmamba/model_state/spatial_classifier_CL_cross_2_layers_best.pth"  # 修改
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载测试数据
    test_dataset = TestDataset("/home/lxz/scmamba/空转/Hubmap_CL_cross/test.csv")  # 修改
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 存储预测结果
    all_preds = []
    all_labels = []
    all_cell_embs = []
    
    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)
            
            # 前向传播
            model_output = model(
                src=src,
                values=values,
                src_key_padding_mask=padding_mask
            )
            
            # 获取预测类别
            preds = torch.argmax(model_output["cls_output"], dim=1)
            cell_embs = model_output["cell_emb"].cpu().numpy()
            
            # 收集结果（过滤未知类型）
            valid_mask = cell_types != -1
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(cell_types[valid_mask].cpu().numpy())
            all_cell_embs.append(cell_embs[valid_mask.cpu().numpy()])  # 只保留有效样本的嵌入
    
    # 拼接所有批次的嵌入
    all_cell_embs = np.concatenate(all_cell_embs, axis=0)  # 形状 [N_valid, 512]
    
    # 保存数据
    np.save("/home/lxz/scmamba/空转/Hubmap_CL_cross/cell_embeddings_test.npy", all_cell_embs)  # 修改
    
    # 保存 labels 和 preds 到一个 .pkl 文件
    with open("/home/lxz/scmamba/空转/Hubmap_CL_cross/labels_preds_test.pkl", "wb") as f:  # 修改
        pickle.dump({
            "labels": all_labels,  # 真实标签
            "preds": all_preds     # 预测标签
        }, f)
    
    # 计算准确率
    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        print(f"\nTest Accuracy: {acc:.4f}")
        
        # 打印分类报告
        from sklearn.metrics import classification_report
        print("\nDetailed Classification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=[id_to_celltype[i] for i in range(class_num)],
            labels=list(range(len(celltype_to_id))),  # 显式指定所有17个类别的标签，测试其它数据时需要更改
            digits=4
        ))
    else:
        print("No valid samples for evaluation")

if __name__ == "__main__":
    evaluate()