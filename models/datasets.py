import os
import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SingleCellDataset(Dataset):
    def __init__(self, h5ad_file, vocab, celltype_mapping, max_seq_len, 
                 gene2id_path=None, batch_size=1000):
        gene2id_path = gene2id_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vocab.json")
        self.vocab = vocab
        self.celltype_mapping = celltype_mapping
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # precompute common values
        self.pad_id = vocab["<pad>"]
        self.cls_id = vocab["<cls>"] 
        
        # more efficient data processing
        self.data = self._process_file_optimized(h5ad_file)
    
    def _process_file_optimized(self, h5ad_file):
        """optimized file-processing"""
        print("Loading data...")
        adata = sc.read(h5ad_file)
        
        # convert to dense matrix (if sparse)
        if sparse.issparse(adata.X):
            print("Converting sparse matrix to dense...")
            expr_matrix = adata.X.toarray()
        else:
            expr_matrix = adata.X
        
        print("Processing cells...")
        
        # indices of valid cells
        valid_cells = []
        cell_types = []
        
        for i in range(adata.n_obs):
            cell_type = adata.obs['annot'].iloc[i]
            if pd.notna(cell_type) and cell_type in self.celltype_mapping:
                valid_cells.append(i)
                cell_types.append(self.celltype_mapping[cell_type])
        
        print(f"Found {len(valid_cells)} valid cells")
        
        # process cells in batches
        data = []
        for batch_start in range(0, len(valid_cells), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(valid_cells))
            batch_indices = valid_cells[batch_start:batch_end]
            batch_celltypes = cell_types[batch_start:batch_end]
            
            batch_data = self._process_cell_batch(
                expr_matrix, batch_indices, batch_celltypes, adata.var_names
            )
            data.extend(batch_data)
            
            if batch_start % (10 * self.batch_size) == 0:
                print(f"Processed {batch_end}/{len(valid_cells)} cells")
        
        return data
    
    def _process_cell_batch(self, expr_matrix, cell_indices, cell_types, gene_names):
        """process cells in batches"""
        batch_data = []
        
        for i, cell_idx in enumerate(cell_indices):
            # expression data of the current cell
            cell_expr = expr_matrix[cell_idx]
            
            # vectorized: find expressed genes
            expressed_mask = cell_expr > 0
            expressed_genes = gene_names[expressed_mask].tolist()
            expr_values = cell_expr[expressed_mask].tolist()
            
            # vectorized ID conversion
            vocab_ids = []
            valid_indices = []
            
            for j, gene in enumerate(expressed_genes):
                gene_id = self.vocab[gene]
                if gene_id is not None:
                    vocab_ids.append(gene_id)
                    valid_indices.append(j)
            
            # keep only valid expression values
            if valid_indices:
                human_values = [expr_values[k] for k in valid_indices]
            else:
                human_values = []
            
            batch_data.append({
                'vocab_ids': vocab_ids,
                'human_values': human_values,
                'celltype': cell_types[i]
            })
        
        return batch_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cell = self.data[idx]
        
        vocab_ids = cell['vocab_ids']
        human_values = cell['human_values']
        
        # length check and handling
        seq_len = len(vocab_ids)
        
        # target length: max_seq_len + 1 (CLS included)
        target_length = self.max_seq_len + 1
        
        if seq_len == 0:
            # handle cells with no expressed genes
            vocab_ids = [self.cls_id] + [self.pad_id] * self.max_seq_len
            human_values = [0.0] + [-2.0] * self.max_seq_len
            padding_mask = [False] + [True] * self.max_seq_len
        else:
            # random truncation if too long
            # at most max_seq_len genes: one slot is reserved for CLS
            if seq_len > self.max_seq_len:
                randidx = np.random.choice(seq_len, self.max_seq_len, replace=False)
                vocab_ids = [vocab_ids[i] for i in randidx]
                human_values = [human_values[i] for i in randidx]
                seq_len = self.max_seq_len
            
            vocab_ids = [self.cls_id] + vocab_ids
            human_values = [0.0] + human_values
            
            # compute padding length
            current_length = len(vocab_ids)  # now seq_len + 1
            padding_length = target_length - current_length  # max_seq_len + 1 - (seq_len + 1) = max_seq_len - seq_len
            
            if padding_length > 0:
                vocab_ids.extend([self.pad_id] * padding_length)
                human_values.extend([-2.0] * padding_length)
            
            # create padding mask
            padding_mask = [vid == self.pad_id for vid in vocab_ids]
        
        # verify consistent lengths
        assert len(vocab_ids) == target_length, f"Length mismatch: expected {target_length}, got {len(vocab_ids)}"
        
        return {
            'src': torch.tensor(vocab_ids, dtype=torch.long),
            'values': torch.tensor(human_values, dtype=torch.float),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'celltype': torch.tensor(cell['celltype'], dtype=torch.long),
        }