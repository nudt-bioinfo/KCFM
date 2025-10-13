"""
数据集：T cancer cell测试数据集
任务：测试微调模型在测试集上的分类性能
输出：分类报告
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import scanpy as sc
import numpy as np
from scipy import sparse
from tqdm import tqdm
import sys
from pathlib import Path
import pickle

# 确保与训练代码相同的配置
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab

# 使用与训练相同的配置参数
batch_size = 32
embsize = 512
nhead = 8
d_hid = 512
nlayers = 6
dropout = 0.1
pad_token = "<pad>"
max_seq_len = 4096
input_emb_style = "continuous"
cell_emb_style = "cls"
pad_value = -2

# 细胞类型映射（必须与训练时一致）
celltype_to_id = {
    'T cell': 0,
    'CD8-positive, alpha-beta cytotoxic T cell': 1,
    'naive thymus-derived CD4-positive, alpha-beta T cell': 2,
    'effector CD8-positive, alpha-beta T cell': 3,
    'effector memory CD8-positive, alpha-beta T cell': 4,
    'central memory CD4-positive, alpha-beta T cell': 5,
    'gamma-delta T cell': 6
}
id_to_celltype = {v: k for k, v in celltype_to_id.items()}
class_num = len(celltype_to_id)

# 加载词汇表（必须与训练时相同）
vocab = GeneVocab.from_file("/home/lxz/scmamba/vocab.json")
ntokens = len(vocab)


# 使用与训练相同的Dataset类
class SingleCellDataset:
    def __init__(self, adata):
        self.adata = adata
        self.cell_ids = adata.obs_names.tolist()
        self.gene_names = adata.var.feature_name.tolist()

        # 预计算非零表达基因的索引
        self.nonzero_indices = {}
        expr_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        for i, cell_id in enumerate(self.cell_ids):
            self.nonzero_indices[cell_id] = np.where(expr_matrix[i] != 0)[0]

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_type = self.adata.obs.loc[cell_id, 'cell_type']

        nonzero_idx = self.nonzero_indices[cell_id]
        expr_values = self.adata.X[idx, nonzero_idx].toarray().flatten() \
            if sparse.issparse(self.adata.X) \
            else self.adata.X[idx, nonzero_idx]
        gene_names = [self.gene_names[i] for i in nonzero_idx]

        gene_ids = []
        filtered_expr = []
        for gene, value in zip(gene_names, expr_values):
            if gene in vocab:
                gene_ids.append(vocab[gene])
                filtered_expr.append(value)

        if len(gene_ids) > max_seq_len - 1:
            selected = np.random.choice(len(gene_ids), max_seq_len - 1, replace=False)
            gene_ids = [gene_ids[i] for i in selected]
            filtered_expr = [filtered_expr[i] for i in selected]

        gene_ids = [vocab["<cls>"]] + gene_ids
        filtered_expr = [0.0] + filtered_expr

        padding_len = max_seq_len - len(gene_ids)
        if padding_len > 0:
            gene_ids += [vocab["<pad>"]] * padding_len
            filtered_expr += [pad_value] * padding_len

        padding_mask = [id_ == vocab["<pad>"] for id_ in gene_ids]

        return {
            'src': torch.LongTensor(gene_ids),
            'values': torch.FloatTensor(filtered_expr),
            'padding_mask': torch.BoolTensor(padding_mask),
            'celltype': torch.tensor(celltype_to_id[cell_type], dtype=torch.long)
        }


def evaluate():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 加载测试数据
    print("加载测试集h5ad文件...")
    adata_test = sc.read("/mnt/HHD16T/DATA/lxz/cancer/T_test.h5ad")
    print(f"测试集加载完成，细胞数量: {adata_test.n_obs}")

    # 创建测试数据集
    test_dataset = SingleCellDataset(adata_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    model = MambaModel(
        ntokens, embsize, nhead, d_hid, nlayers,
        vocab=vocab, dropout=dropout, pad_token=pad_token,
        pad_value=pad_value, input_emb_style=input_emb_style,
        cell_emb_style=cell_emb_style, class_num=class_num
    ).to(device)

    # 加载训练好的模型权重
    model_path = '/home/lxz/scmamba/model_state/cancer_Tcell_final_2_layers_10_epochs.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型权重已加载: {model_path}")

    # 初始化收集变量
    all_preds = []
    all_labels = []
    all_cell_embs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
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

            # 获取预测类别和细胞嵌入
            preds = torch.argmax(model_output["cls_output"], dim=1)
            cell_embs = model_output["cell_emb"].cpu().numpy()

            # 收集结果（这里假设不需要过滤未知类型，因为测试集应该都是已知类型）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cell_types.cpu().numpy())
            all_cell_embs.append(cell_embs)

    # 拼接所有批次的嵌入
    all_cell_embs = np.concatenate(all_cell_embs, axis=0)  # 形状 [N, 512]

    # 保存细胞嵌入
    embedding_path = "/home/lxz/scmamba/aicase1/cell_embeddings.npy"
    np.save(embedding_path, all_cell_embs)
    print(f"细胞嵌入已保存到: {embedding_path}")

    # 保存 labels 和 preds 到一个 .pkl 文件
    pkl_path = "/home/lxz/scmamba/aicase1/labels_preds.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "labels": all_labels,  # 真实标签
            "preds": all_preds  # 预测标签
        }, f)
    print(f"标签和预测结果已保存到: {pkl_path}")

    # 计算准确率
    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n测试准确率: {acc:.4f}")

        # 打印分类报告
        print("\n详细分类报告:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=list(celltype_to_id.keys()),
            digits=4
        ))

    #     # 可选：将分类报告保存到文件
    #     report = classification_report(
    #         all_labels,
    #         all_preds,
    #         target_names=list(celltype_to_id.keys()),
    #         digits=4,
    #         output_dict=True
    #     )
    #     report_path = "/home/lxz/scmamba/空转/Hubmap_CL_cross/classification_report.pkl"
    #     with open(report_path, "wb") as f:
    #         pickle.dump(report, f)
    #     print(f"分类报告已保存到: {report_path}")
    # else:
    #     print("没有有效样本可用于评估")


if __name__ == "__main__":
    evaluate()