import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np
from scipy import sparse
from config.Bcell_config import DATA_CONFIG, CELLTYPE_MAPPING
#from config.ann3_config import DATA_CONFIG, CELLTYPE_MAPPING
class SingleCellDataset(Dataset):
    def __init__(self, h5ad_file, vocab, max_seq_len):
        self.data = []
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self._process_file(h5ad_file)
        
    def _process_file(self, h5ad_file):
        """处理单个h5ad文件"""
        adata = sc.read(h5ad_file)
        expr_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        
        for i in range(adata.n_obs):
            #cell_type = adata.obs['ann3'].iloc[i]
            cell_type = adata.obs['clusters'].iloc[i]
            #这里会自动过滤掉nan值
            if cell_type not in CELLTYPE_MAPPING:
                continue
                
            # 获取基因表达对
            expressed_genes = []
            expr_values = []
            for j, gene in enumerate(adata.var_names):
                val = expr_matrix[i, j]
                if val > 0:  # 只保留有表达的基因
                    expressed_genes.append(gene)
                    expr_values.append(val)
            
            # 转换为vocab ID
            vocab_ids = [self.vocab[gene] if gene in self.vocab else None for gene in expressed_genes]
            human_values = expr_values
            
            self.data.append({
                'vocab_ids': vocab_ids,
                'human_values': human_values,
                'celltype': CELLTYPE_MAPPING[cell_type]
            })
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cell = self.data[idx]
        
        # 获取基因ID和表达值
        vocab_ids = cell['vocab_ids']
        human_values = cell['human_values']
        
        # 过滤无效数据
        valid_indices = [i for i, x in enumerate(vocab_ids) if x is not None]
        vocab_ids = [vocab_ids[i] for i in valid_indices]
        human_values = [human_values[i] for i in valid_indices]
        
        # 随机截取
        if len(vocab_ids) > self.max_seq_len:
            randidx = np.random.choice(len(vocab_ids), self.max_seq_len, replace=False)
            vocab_ids = [vocab_ids[i] for i in randidx]
            human_values = [human_values[i] for i in randidx]
        
        # 填充序列
        padding_length = self.max_seq_len - len(vocab_ids)
        if padding_length > 0:
            vocab_ids = vocab_ids + [self.vocab[DATA_CONFIG["pad_token"]]] * padding_length
            human_values = human_values + [DATA_CONFIG["pad_value"]] * padding_length
        
        # 添加CLS token
        vocab_ids = [self.vocab["<cls>"]] + vocab_ids
        human_values = [0.0] + human_values
       
        # 创建padding mask
        padding_mask = [False] * len(vocab_ids)
        for i in range(len(vocab_ids)):
            if vocab_ids[i] == self.vocab[DATA_CONFIG["pad_token"]]:
                padding_mask[i] = True
        
        return {
            'src': torch.tensor(vocab_ids, dtype=torch.long),
            'values': torch.tensor(human_values, dtype=torch.float),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'celltype': torch.tensor(cell['celltype'], dtype=torch.long),
        }