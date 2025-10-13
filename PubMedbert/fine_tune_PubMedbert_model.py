"""
使用知识图谱中的细胞类型微调bert模型
"""
import re
from torch import nn
from torch.optim import AdamW  
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F  
def extract_triplets_and_types(file_path):
    """
    提取文件中所有三元组和唯一细胞类型
    返回:
        triplets: [(head, relation, tail), ...]
        cell_types: set() 所有唯一的细胞类型
    """
    triplets = []
    cell_types = set()
    
    # 匹配所有预定义关系（使用正则表达式中的命名捕获组）
    pattern = re.compile(
    r"(?P<head>.+?)\s+"
    r"(?P<relation>is_a|disjoint_from|exact_synonyms|broad_synonyms|"
    r"related_synonyms|develops_from|develops_into|synapsed_to)\s+"
    r"(?P<tail>.+)"
)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 非空行处理
                match = pattern.match(line)
                if match:
                    head = match.group('head')
                    relation = match.group('relation')
                    tail = match.group('tail')
                    triplets.append((head, relation, tail))
                    cell_types.update([head, tail])
    
    return triplets, cell_types





file_path = "/home/lxz/PubMedbert/triples.txt"
triplets, cell_types = extract_triplets_and_types(file_path)
batch_size=64
# 定义关系映射字典
relation_mapping = {
    "is_a": 0,
    "disjoint_from": 1,
    "exact_synonyms": 2,
    "broad_synonyms": 3,
    "related_synonyms": 4,
    "develops_from": 5,
    "develops_into": 6,
    "synapsed_to": 7
}
# 转换为描述对 + 关系ID
data = []
for h, r, t in triplets:
    data.append({
        "head": h,
        "tail": t,
        "relation": relation_mapping[r]  # 将关系转为数字ID
    })

"""
| is_a                                   | 3818 |
| disjoint_from                          | 32   |
| exact_synonyms                         | 2270 |
| broad_synonyms                         | 262  |
| related_synonyms                       | 527  |
| develops_from                          | 373  |
| develops_into                          | 14   |
| synapsed_to                            | 27   |
"""





def mean_pooling(output, mask):#标记哪些是真实token（值为1），哪些是填充的padding token（值为0）
    embeddings = output[0]  # 所有Token的Embedding，形状为 (batch_size, sequence_length, hidden_size)
    # print(embeddings.shape)
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)



class TripletPubMedBERT(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 加载BERT模型和tokenizer
        self.bert = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 添加投影层 (768->512),scgpt的是512,mamba是512,geneformer是256,sctab是128,scfoundation是768
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1)  
        )
        
        # 关系投影层（保持维度匹配）
        self.rel_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1) 
        )
        # 定义关系列表（必须与relation_mapping顺序一致）
        self.relation_names = [
            "is_a", "disjoint_from", "exact_synonyms",  
            "broad_synonyms", "related_synonyms",
            "develops_from", "develops_into", "synapsed_to"
        ]
        
        # 用PubMedBERT初始化并冻结关系嵌入
        with torch.no_grad():
            # 编码所有关系名称
            inputs = self.tokenizer(
                self.relation_names,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            outputs = self.bert(**inputs)
            # 对每个关系名称取平均作为初始向量
            relation_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # 创建冻结的关系嵌入层
        self.rel_emb = nn.Embedding.from_pretrained(
            relation_embeddings,
            freeze=False  # 冻结参数
        )
        
        # 冻结BERT的前3层
        for param in self.bert.encoder.layer[:3].parameters():
            param.requires_grad = False

    def forward(self, head, tail, relation):
        # 编码头尾实体
        h_out = self.bert(**head)
        t_out = self.bert(**tail)
        
        # 均值池化
        h_pool = mean_pooling(h_out, head['attention_mask'])
        t_pool = mean_pooling(t_out, tail['attention_mask'])
        h_pool = self.projection(h_pool)
        t_pool = self.projection(t_pool)
        r_vec = self.rel_projection(self.rel_emb(relation))  
        
        return h_pool, t_pool, r_vec



class BioDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "head": item["head"],
            "tail": item["tail"],
            "relation": torch.tensor(item["relation"], dtype=torch.long)
        }
class BioTripletLoss(nn.Module):
    def __init__(self, margin=0.3, min_pos_dist=0.1, push_scale=2.0, 
                 dissimilar_relation_ids=None, hard_neg_ratio=0.3):
        super().__init__()
        self.margin = margin
        self.min_pos_dist = min_pos_dist
        self.push_scale = push_scale
        self.dissimilar_relation_ids = dissimilar_relation_ids or {1}
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, h, t, r, relation_ids):
        loss = 0
        batch_size = h.size(0)
        
        for i in range(batch_size):
            rel_id = relation_ids[i].item()
            pos_dist = torch.norm(h[i] + r[i] - t[i], p=2)
            
            if rel_id in self.dissimilar_relation_ids:
                # 推远关系
                curr_margin = self.margin * self.push_scale
                loss += torch.relu(curr_margin - pos_dist) + 0.5 * torch.exp(-pos_dist)
            else:
                # # 拉近关系
                # if torch.rand(1, device=h.device) < self.hard_neg_ratio:
                #     # 困难负采样
                #     with torch.no_grad():
                #         sim = F.cosine_similarity(h[i]+r[i], t, dim=-1)
                #         sim[i] = -float('inf')  # 排除自己
                #         neg_idx = torch.argmax(sim).item()
                # else:
                    # 随机负采样
                neg_idx = torch.randint(0, batch_size, (1,), device=h.device).item()
                while neg_idx == i:
                    neg_idx = torch.randint(0, batch_size, (1,), device=h.device).item()
                
                neg_dist = torch.norm(h[i] + r[i] - t[neg_idx], p=2)
                loss += torch.relu(pos_dist - neg_dist + self.margin) + \
                        0.3 * torch.relu(self.min_pos_dist - pos_dist)
                
        return loss / batch_size
def collate_fn(batch):
    heads = tokenizer([x["head"] for x in batch], padding=True, truncation=True, max_length=25,return_tensors="pt")
    tails = tokenizer([x["tail"] for x in batch], padding=True, truncation=True, max_length=25,return_tensors="pt")
    relations = torch.stack([x["relation"] for x in batch])
    return heads, tails, relations



model_path = "/home/lxz/PubMedbert"   
model = TripletPubMedBERT(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 数据加载
dataset = BioDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# 训练配置
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = BioTripletLoss(margin=0.5,min_pos_dist=0.1)


# 初始化记录列表
train_losses = []

for epoch in range(20):
    epoch_losses = []  # 记录当前epoch的所有batch损失
    
    for heads, tails, rels in loader:
        # 数据移动到GPU
        heads = {k: v.to("cuda") for k, v in heads.items()}
        tails = {k: v.to("cuda") for k, v in tails.items()}
        rels = rels.to("cuda")
        
        # 前向传播
        h, t, r = model(heads, tails, rels)
        loss = loss_fn(h, t, r, rels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录当前batch的损失
        batch_loss = loss.item()
        epoch_losses.append(batch_loss)
        print(f"Epoch {epoch}, Batch Loss: {batch_loss:.6f}")
    
    # 计算并记录当前epoch的平均损失
    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(epoch_avg_loss)
    print(f"Epoch {epoch} completed. Average Loss: {epoch_avg_loss:.6f}")

# 保存模型
output_dir = "/home/lxz/PubMedbert/"
os.makedirs(output_dir, exist_ok=True)

torch.save({
    'bert_state': model.bert.state_dict(),
    'projection_state': model.projection.state_dict(),
    'rel_projection_state': model.rel_projection.state_dict()
}, f"{output_dir}/fine_tune_bert_768_dim.pth")


# torch.save({
#     'bert_state': model.bert.state_dict()
# }, f"{output_dir}/finetune_bert.pth")
# 加载时需要重建完整架构
# def load_model(model_path, save_path):
#     model = TripletPubMedBERT(model_path).to("cuda")
#     checkpoint = torch.load(save_path)
    
#     model.bert.load_state_dict(checkpoint['bert_state'])
#     model.projection.load_state_dict(checkpoint['projection_state'])
#     model.rel_projection.load_state_dict(checkpoint['rel_projection_state'])
    
#     return model
# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# 保存图像
plt.savefig(f"{output_dir}/training_loss_768_dim.png")
plt.show()




