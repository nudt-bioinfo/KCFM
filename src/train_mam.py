import os
import sys
from collections import defaultdict
import warnings
from os.path import join
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import random
sys.path.insert(0, "../")
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
os.environ["TRITON_AUTOTUNE"] = "0"

# 配置参数
mask_ratio = 0.15
epochs = 1
embsize = 512
nhead = 8
d_hid = 512
nlayers = 6
dropout = 0.1
lr = 5e-5
pad_token = "<pad>"
max_seq_len = 2049
input_style = "binned"
input_emb_style = "continuous"
cell_emb_style = "cls"
vocab = GeneVocab.from_file("/home/lxz/scgpt-test/scGPT_human/vocab.json")
ntokens = len(vocab)
mask_value = -1
pad_value = -2

# 明确指定使用GPU 0和1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 关系图构建函数
def build_relation_graph(triplets_file):
    """
    构建细胞类型关系图
    返回:
        dict: {
            celltype: {
                'close': set(),  # 需要拉近的类型
                'far': set()     # 需要拉远的类型
            }
        }
    """
    graph = defaultdict(lambda: {'close': set(), 'far': set()})
    
    with open(triplets_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 3:
                continue
                
            h, r, t = parts[0], parts[1], ' '.join(parts[2:])
            
            if r in ["is_a", "exact_synonyms", "develops_from", "develops_into"]:
                graph[h]['close'].add(t)
                graph[t]['close'].add(h)
            elif r == "disjoint_from":
                graph[h]['far'].add(t)
                graph[t]['far'].add(h)
    
    return graph
# 加载关系图

relation_graph = build_relation_graph("/home/lxz/PubMedbert/triples.txt")
def get_contrastive_pairs(cell_type, all_cell_types):
    """
    获取对比学习需要的正负样本类型
    参数:
        cell_type: 目标细胞类型
        all_cell_types: 所有细胞类型的列表
    返回:
        (close_type, far_type)
    """
    relations = relation_graph.get(cell_type, {'close': set(), 'far': set()})
    
    # 选择需要拉近的类型
    close_candidates = list(relations['close']) if relations['close'] else all_cell_types
    close_type = random.choice(close_candidates)
    
    # 选择需要拉远的类型
    far_candidates = list(relations['far']) if relations['far'] else [
        t for t in all_cell_types if t != cell_type and t not in relations['close']
    ]
    far_type = random.choice(far_candidates) if far_candidates else random.choice(all_cell_types)
    return close_type, far_type
def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def hierarchical_ranking_loss(
    cell_emb, 
    cell_type_ids,
    type_embeddings,
    relation_graph,
    all_cell_types,
    margin1=0.1,  # 自身与close的间距
    margin2=0.2   # close与其他类型的间距
):
    """
    层级排序对比损失
    
    输入参数与前文一致，新增：
        margin1: 自身类型与close类型的最小间隔
        margin2: close类型与其他类型的最小间隔
    """
    # 归一化处理
    cell_emb = F.normalize(cell_emb, p=2, dim=1)
    type_embeddings = F.normalize(type_embeddings, p=2, dim=1)
    batch_size = cell_emb.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        anchor = cell_emb[i]
        current_type = all_cell_types[cell_type_ids[i].item()]
        relations = relation_graph.get(current_type, {'close': set(), 'far': set()})
        
        #############################################
        # Level 1: 自身相似性必须最高
        #############################################
        self_emb = type_embeddings[cell_type_ids[i]]
        self_sim = F.cosine_similarity(anchor, self_emb, dim=0)
        
        #############################################
        # Level 2: 收集所有close类型嵌入
        #############################################
        close_embs = []
        for t in relations['close']:
            if t in all_cell_types:
                close_embs.append(type_embeddings[all_cell_types.index(t)])
        if not close_embs:  # 若无close类型，随机选一个非自身类型作为proxy
            proxy_idx = random.choice([idx for idx in range(len(all_cell_types)) if idx != cell_type_ids[i]])
            close_embs.append(type_embeddings[proxy_idx])
        close_embs = torch.stack(close_embs)
        
        #############################################
        # Level 3: 采样负样本（非自身且非close类型）
        #############################################
        negative_candidates = [
            idx for idx, t in enumerate(all_cell_types) 
            if t != current_type and t not in relations['close']
        ]
        if not negative_candidates:  # 保底机制
            negative_candidates = [idx for idx in range(len(all_cell_types)) if idx != cell_type_ids[i]]
        neg_idx = random.choice(negative_candidates)
        neg_emb = type_embeddings[neg_idx]
        
        #############################################
        # 计算层级排序损失
        #############################################
        # 计算所有close类型与anchor的相似度
        close_sims = F.cosine_similarity(anchor.unsqueeze(0), close_embs)
        max_close_sim = torch.max(close_sims)
        
        # 计算负样本相似度
        neg_sim = F.cosine_similarity(anchor, neg_emb, dim=0)
        
        # Level1-2 约束：自身 > 最相似的close类型 + margin1
        loss_self_vs_close = F.relu(max_close_sim - self_sim + margin1)
        
        # Level2-3 约束：最相似的close类型 > 负样本 + margin2 
        loss_close_vs_neg = F.relu(neg_sim - max_close_sim + margin2)
        
        total_loss += loss_self_vs_close + loss_close_vs_neg

    return total_loss / batch_size  # 返回批次平均损失
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare_model(rank, model):
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    return model

def prepare_dataloader(dataset, rank, world_size, batch_size=32, num_workers=0):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class finuetune_BERT(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0)  
        )
        
    def forward(self, **inputs):
        outputs = self.bert(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.projection(pooled)
    
def get_embedding(model, tokenizer, text, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=25).to(device)
    with torch.no_grad():
        output = model(**inputs)
        return output.cpu().numpy()

def generate_masked_positions(padding_mask, mask_prob=0.15, device="cuda"):
    mask = torch.zeros_like(padding_mask, dtype=torch.float32, device=device)
    for i in range(padding_mask.size(0)):
        non_padding = torch.where(~padding_mask[i])[0]
        n_mask = int(len(non_padding) * mask_prob)
        masked_idx = torch.randperm(len(non_padding), device=device)[:n_mask]
        mask[i, non_padding[masked_idx]] = 1.0
    return mask.bool()
def masked_mse_loss(input, target, mask):
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / (mask.sum() + 1e-8)

class RankSplitParquetDataset(Dataset):
    def __init__(self, path: str, rank: int = 0, world_size: int = 1, 
                 start_file: str = None, end_file: str = None):
        """
        按GPU进程分配独立文件的数据集类 (奇偶分配)
        
        参数:
            path: 数据目录路径
            rank: 当前GPU进程编号 (0, 1, ...)
            world_size: 总GPU数量
            start_file: 起始文件名 (如"part.0.parquet")
            end_file: 结束文件名 (如"part.100.parquet")
        """
        self.path = path
        self.rank = rank
        self.world_size = world_size
        
        # 获取所有文件并按数字排序
        self.all_files = sorted(
            [f for f in os.listdir(path) if f.endswith('.parquet')],
            key=lambda x: int(x.split('.')[1])
        )
        
        # 应用起始和结束文件过滤
        if start_file or end_file:
            start_idx = 0
            end_idx = len(self.all_files)
            
            if start_file:
                start_idx = next((i for i, f in enumerate(self.all_files) if f == start_file), 0)
            if end_file:
                end_idx = next((i for i, f in enumerate(self.all_files) if f == end_file), len(self.all_files)) + 1
            
            self.all_files = self.all_files[start_idx:end_idx]
        
        # 为当前rank分配文件 (奇偶分配)
        self.assigned_files = self._assign_files_odd_even()
        print(f"Rank {rank} assigned {len(self.assigned_files)} files: {[f.split('/')[-1] for f in self.assigned_files[:3]]}...")
        
        # 初始化状态
        self.current_file_idx = 0
        self.current_row_idx = 0
        self.current_df = None
        self.current_file_name = None
        self._load_current_file()

    def _assign_files_odd_even(self):
        """按奇偶分配文件: rank 0取偶数文件, rank 1取奇数文件"""
        if self.world_size != 2:
            raise ValueError("奇偶分配仅支持world_size=2的情况")
        
        # 按文件编号的奇偶性分配
        assigned_files = []
        for i, f in enumerate(self.all_files):
            file_num = int(f.split('.')[1])
            if (file_num % 2 == self.rank):  # rank 0取偶数，rank 1取奇数
                assigned_files.append(os.path.join(self.path, f))
        
        return assigned_files

    def _load_current_file(self):
        """加载当前文件并打印日志"""
        if self.current_file_idx < len(self.assigned_files):
            file_path = self.assigned_files[self.current_file_idx]
            self.current_file_name = os.path.basename(file_path)
            self.current_df = pq.read_table(file_path).to_pandas()
            self.current_row_idx = 0  # 重置行索引
            print(f"Rank {self.rank} 开始训练文件: {self.current_file_name} (总行数: {len(self.current_df)})")
        else:
            self.current_df = None
            self.current_file_name = None

    def __len__(self):
        """返回当前rank分配到的总数据量"""
        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.assigned_files)

    def __getitem__(self, _):
        """
        获取下一个数据项（忽略输入索引，使用内部状态管理）
        返回:
            dict: {'X': tensor, 'cell_type': int}
        """
        while True:
            # 如果当前文件已读完，加载下一个文件
            if self.current_df is None or self.current_row_idx >= len(self.current_df):
                # 打印当前文件完成信息
                if self.current_file_name:
                    print(f"Rank {self.rank} 完成文件: {self.current_file_name} (已处理行数: {self.current_row_idx})")
                
                self.current_file_idx += 1
                self._load_current_file()
                if self.current_df is None:  # 所有文件已读完
                    raise StopIteration
                continue
            
            # 返回当前数据
            row = self.current_df.iloc[self.current_row_idx]
            self.current_row_idx += 1
            
            
            # if self.current_row_idx % 1000 == 0:
            #     print(f"Rank {self.rank} 正在处理文件 {self.current_file_name} - 进度: {self.current_row_idx}/{len(self.current_df)}")
            
            return {
                'X': torch.FloatTensor(row['X']),
                'cell_type': row['cell_type']
            }

def train(rank, world_size):
    # 初始化分布式训练
    setup(rank, world_size) 
    
    # 设备设置
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    with torch.cuda.device(device):
        # 初始化模型
        
        model = MambaModel(
            ntokens, embsize, nhead, d_hid, nlayers,
            vocab=vocab, dropout=dropout, pad_token=pad_token,
            pad_value=pad_value, input_emb_style=input_emb_style,
            cell_emb_style=cell_emb_style
        ).to(device)
        
        # # 加载预训练权重
        model_dict = model.state_dict()
        pretrained_dict = torch.load("/home/lxz/scmamba/model_state/cell_cls_3loss_6layer1.pth", map_location=f"cuda:{rank}")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 冻结加载的参数
        # for name, param in model.named_parameters():
        #     if name in pretrained_dict:
        #         param.requires_grad = False

        # # 打印哪些参数需要训练，哪些不需要
        # print("需要训练的参数：")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"  {name}")
        # 准备DDP模型
        model = prepare_model(rank, model)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # 加载BERT模型生成celltype embedding
        tokenizer = AutoTokenizer.from_pretrained("/home/lxz/PubMedbert")
        finetuned_model = finuetune_BERT(AutoModel.from_pretrained("/home/lxz/PubMedbert")).to(rank)
        checkpoint = torch.load("/home/lxz/PubMedbert/finetune_bert.pth", map_location=f"cuda:{rank}")
        finetuned_model.bert.load_state_dict(checkpoint['bert_state'])
        finetuned_model.projection.load_state_dict(checkpoint['projection_state'])
        
        # 生成celltype embedding
        df = pd.read_parquet("/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/categorical_lookup/cell_type.parquet")
        celltype_emb = []
        for cell_type in tqdm(df['label']):
            emb = get_embedding(finetuned_model, tokenizer, cell_type, f"cuda:{rank}")
            celltype_emb.append(emb.squeeze(0))
        celltype_emb = np.stack(celltype_emb)
        celltype_emb_tensor = torch.from_numpy(celltype_emb).to(rank)
        all_cell_types = df['label'].tolist()
        
        # 加载基因名称
        df = pd.read_parquet('/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/var.parquet')
        feature_names = df['feature_name'].tolist()
    
    # 数据加载
    dataset = RankSplitParquetDataset(
        "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/test",
        rank=rank,
        world_size=world_size,
        start_file="part.30.parquet",
        end_file="part.40.parquet"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 数据预处理
                x = batch['X'].to(rank)
                cell_types = batch['cell_type'].to(rank)
                
                batch_values = []
                batch_src = []
                batch_padding_mask = []
                for cell_expr in x:
                    cell_expr = cell_expr.to(rank)
                    nonzero_idx = torch.nonzero(cell_expr).squeeze().to(rank)
                    nonzero_values = cell_expr[nonzero_idx]
                    
                    if len(nonzero_values) < max_seq_len - 1:
                        cls_value = torch.tensor([0.0], device=f"cuda:{rank}")
                        padding = torch.full((max_seq_len - 1 - len(nonzero_values),), -2.0, device=f"cuda:{rank}")
                        values = torch.cat([cls_value, nonzero_values, padding])
                        
                        cls_idx = torch.tensor([-1], device=f"cuda:{rank}")
                        padding_idx = torch.full((max_seq_len - 1 - len(nonzero_idx),), -1, device=f"cuda:{rank}")
                        extended_idx = torch.cat([cls_idx, nonzero_idx, padding_idx])
                    else:
                        sampled_idx = torch.randperm(len(nonzero_idx))[:max_seq_len - 1]
                        extended_idx = nonzero_idx[sampled_idx]
                        values = torch.cat([
                            torch.tensor([0.0], device=f"cuda:{rank}"),
                            nonzero_values[sampled_idx]
                        ])

                    batch_values.append(values)

                    # 处理基因名称
                    valid_idx = extended_idx[extended_idx >= 0]
                    cell_genes = [feature_names[i] for i in valid_idx.cpu().numpy()]
                    cell_genes = ["<cls>"] + cell_genes
                    # print(cell_genes) #['<cls>', 'SKI', 'CAMTA1', 'H6PD', 'CTNNBIP1', 'MTHFR', 'PRDM2', 'SZRD1'
                    # print(len(cell_genes))#1004
                    if len(cell_genes) < max_seq_len:
                        cell_genes += ["<pad>"] * (max_seq_len - len(cell_genes))

                    padding_mask = [gene == "<pad>" for gene in cell_genes]
                    batch_padding_mask.append(padding_mask)
                    
                    src = [vocab[gene] if gene in vocab else vocab["<unk>"] for gene in cell_genes]
                    # print(src)#[60695, 31254, 3896, 10778, 7131, 17121, 20235, 32770, 17611,
                    # print(len(src))#2049
                    batch_src.append(src)

                # 转换为tensor
                values_tensor = torch.stack(batch_values)
                src_tensor = torch.tensor(batch_src, device=f"cuda:{rank}")
                src_key_padding_mask = torch.tensor(batch_padding_mask, dtype=torch.bool, device=f"cuda:{rank}")
                #print(src_key_padding_mask)[False, False, False,  ...,  True,  True,  True]
                #print(src_key_padding_mask.shape)torch.Size([48, 2049])
                # 生成MLM的mask位置
                masked_positions = generate_masked_positions(src_key_padding_mask, mask_prob=0.15, device=f"cuda:{rank}")
                values_tensor[masked_positions] = -1.0

                # 前向传播
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):
                    model_output = model(
                        src=src_tensor,
                        values=values_tensor,
                        src_key_padding_mask=src_key_padding_mask
                    )
                    
                 

                    # 计算损失
                    loss_mse = masked_mse_loss(model_output["mlm_output"], values_tensor, masked_positions)
                    loss_cls = criterion_cls(model_output["cls_output"], cell_types)
                    loss_relation = hierarchical_ranking_loss(
                        cell_emb=model_output["cell_emb"],
                        cell_type_ids=cell_types,
                        type_embeddings=celltype_emb_tensor,
                        relation_graph = build_relation_graph("/home/lxz/PubMedbert/triples.txt"),
                        all_cell_types=all_cell_types,
                    )
                    loss = 0.5 * loss_mse + 1.0 * loss_cls + 0.5 * loss_relation
                    # loss = 0.5 * loss_mse + 1.0 * loss_cls 
              

                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

                # 日志记录
                if rank == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Total Loss: {loss.item():.4f}, "
                        f"MLM Loss: {loss_mse.item():.4f}, "
                        f"Cls Loss: {loss_cls.item():.4f}, "
                        f"Relation Loss: {loss_relation:.4f}"
                    )
                    global_step = epoch * len(dataloader) + batch_idx
                   

                # 定期保存模型
                if batch_idx % 500 == 0 and rank == 0 and batch_idx!=0:
                    torch.save(model.module.state_dict(), '/home/lxz/scmamba/model_state/cell_cls_3loss_6layer1.pth')

            except Exception as e:
                print(f"Rank {rank}: Error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue


    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )






     
    # binning=51
    # for epoch in range(epochs):
    #     for batch_idx, batch in enumerate(dataloader):
    #         x = batch['X'].to(rank)
    #         cell_types = batch['cell_type'].to(rank)
            
    #         # 处理batch数据
    #         batch_values = []
    #         batch_src = []
    #         batch_padding_mask = []
    #         for cell_expr in x:
    #             # 分箱处理
    #             # if binning:
    #             if not isinstance(binning, int):
    #                 raise ValueError("Binning arg must be an integer, but got {}.".format(binning))
    #             n_bins = binning  # NOTE: the first bin is always a special for zero
    #             nonzero_idx = torch.nonzero(cell_expr).squeeze()
    #             nonzero_values = cell_expr[nonzero_idx]
    #             bins = torch.quantile(nonzero_values, torch.linspace(0, 1, n_bins - 1, device=nonzero_values.device))
    #             binned_nonzero_values = torch.bucketize(nonzero_values, bins, right=True)
    #             binned_values = torch.zeros_like(cell_expr, dtype=torch.int64)
    #             binned_values[nonzero_idx] = binned_nonzero_values
    #             # else:
    #             #     binned_values = cell_expr  # 如果没有启用分箱，直接使用原始值
                
    #             # 处理分箱后的值
    #             nonzero_idx = torch.nonzero(binned_values).squeeze()
    #             nonzero_values = binned_values[nonzero_idx]
                
    #             if len(nonzero_values) < max_seq_len - 1:
    #                 cls_value = torch.tensor([0], device=f"cuda:{rank}")
    #                 padding = torch.full((max_seq_len - 1 - len(nonzero_values),), -2, device=f"cuda:{rank}")
    #                 values = torch.cat([cls_value, nonzero_values, padding])
                    
    #                 cls_idx = torch.tensor([-1], device=f"cuda:{rank}")
    #                 padding_idx = torch.full((max_seq_len - 1 - len(nonzero_idx),), -1, device=f"cuda:{rank}")
    #                 extended_idx = torch.cat([cls_idx, nonzero_idx, padding_idx])
    #             else:
    #                 sampled_idx = torch.randperm(len(nonzero_idx))[:max_seq_len - 1]
    #                 extended_idx = nonzero_idx[sampled_idx]
    #                 values = torch.cat([
    #                     torch.tensor([0], device=f"cuda:{rank}"),
    #                     nonzero_values[sampled_idx]
    #                 ])
    #             values = values.to(dtype=torch.float32)# 或 torch.half，根据模型权重的数据类型选择
    #             batch_values.append(values)

    #             # 处理基因名称
    #             valid_idx = extended_idx[extended_idx >= 0]
    #             cell_genes = [feature_names[i] for i in valid_idx.cpu().numpy()]
    #             cell_genes = ["<cls>"] + cell_genes
    #             if len(cell_genes) < max_seq_len:
    #                 cell_genes += ["<pad>"] * (max_seq_len - len(cell_genes))

    #             padding_mask = [gene == "<pad>" for gene in cell_genes]
    #             batch_padding_mask.append(padding_mask)
                
    #             src = [vocab[gene] if gene in vocab else vocab["<unk>"] for gene in cell_genes]
    #             batch_src.append(src)
            
    #         # 转换为tensor
    #         values_tensor = torch.stack(batch_values)
    #         src_tensor = torch.tensor(batch_src, device=f"cuda:{rank}")
    #         src_key_padding_mask = torch.tensor(batch_padding_mask, dtype=torch.bool, device=f"cuda:{rank}")
            
    #         # 生成MLM的mask位置
    #         masked_positions = generate_masked_positions(src_key_padding_mask, mask_prob=0)
    #         values_tensor[masked_positions] = -1.0

    #         # 前向传播
    #         optimizer.zero_grad()
    #         with torch.cuda.amp.autocast(enabled=False):
    #             model_output = model(
    #                 src=src_tensor,
    #                 values=values_tensor,
    #                 src_key_padding_mask=src_key_padding_mask
    #             )
                