# import torch
# from transformers import AutoTokenizer, AutoModel

# # 加载配置
# checkpoint = torch.load(
#     "/home/lxz/PubMedbert_finetuned/full_model.pth", 
#     weights_only=True  # 启用安全模式
# )
# model_path = checkpoint['config']['model_path']

# # 初始化原始模型和自定义模型
# tokenizer = AutoTokenizer.from_pretrained("/home/lxz/PubMedbert_finetuned/")  # 加载保存的tokenizer
# bert_model = AutoModel.from_pretrained(model_path)

# # 重建完整模型
# class TripletPubMedBERT(torch.nn.Module):
#     def __init__(self, bert_model):
#         super().__init__()
#         self.bert = bert_model
#         self.rel_emb = torch.nn.Embedding(8, 768)
        
# model = TripletPubMedBERT(bert_model).to("cuda")
# model.load_state_dict(checkpoint['bert_state'], strict=False)  # 加载BERT部分
# model.rel_emb.load_state_dict(checkpoint['rel_emb_state'])  # 加载关系嵌入层

# def get_word_embedding(word):
#     inputs = tokenizer(word, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         outputs = model.bert(**inputs)
#         # 使用均值池化
#         mask = inputs['attention_mask'].unsqueeze(-1)
#         embeddings = outputs.last_hidden_state * mask
#         pooled = embeddings.sum(dim=1) / mask.sum(dim=1)
#     return pooled.cpu().numpy()

# # 示例：获取"neuron"的embedding
# embedding = get_word_embedding("neuron")
# print(embedding)
# print(embedding.shape)  # 应输出 (1, 768)




from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 配置设置
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = "/home/lxz/PubMedbert"
finetuned_path = "/home/lxz/PubMedbert_finetuned/full_model.pth"

# 加载基础模型和tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModel.from_pretrained(base_model_path).to(device)

# 定义微调模型结构
class FineTunedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model  # 直接使用传入的模型实例
        self.rel_emb = torch.nn.Embedding(8, 768)
        
    def forward(self, **inputs):
        return self.bert(**inputs)

# 加载微调模型（关键修复点）
finetuned_model = FineTunedModel(base_model).to(device)  # 共享基础模型结构

# 调试：检查checkpoint内容
checkpoint = torch.load(finetuned_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())  # 确认包含'bert_state'

# 修正键名映射（根据实际checkpoint调整）
adjusted_state_dict = {
    k.replace('module.bert.', 'bert.').replace('bert.', ''): v 
    for k, v in checkpoint['bert_state'].items()
}

# 严格加载参数并检查缺失键
load_result = finetuned_model.load_state_dict(adjusted_state_dict, strict=False)
print("\n参数加载情况:")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

# 验证参数差异（关键步骤）
print("\n参数差异验证:")
with torch.no_grad():
    # 随机选取10个参数比较
    for name, param in base_model.named_parameters():
        if name in adjusted_state_dict:
            diff = (param - adjusted_state_dict[name]).abs().max().item()
            if diff > 1e-6:
                print(f"🔴 {name}: 存在差异 (max_diff={diff:.6f})")
            else:
                print(f"🟢 {name}: 无差异")
        else:
            print(f"⚠️ {name} 不在微调模型中")

# 均值池化函数
def mean_pooling(output, mask):
    embeddings = output.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# 测试术语对
term_pairs = [
    ("Primary cultured cells", "primary cultured cell"),
    ("Primary cultured cells", "neural crest derived fibroblast"),
    ("T cell", "B cell"),
    ("neuron", "glial cell"),
    ("macrophage", "dendritic cell"),
    ("erythrocyte", "keratinocyte")
]

# 计算相似度
def compute_similarity(model, tokenizer, text_pair):
    inputs = tokenizer(text_pair, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

# 结果收集
results = []
for pair in term_pairs:
    base_sim = compute_similarity(base_model, base_tokenizer, pair)
    ft_sim = compute_similarity(finetuned_model, base_tokenizer, pair)  # 使用完整微调模型
    
    results.append({
        "term_pair": pair,
        "base_model": base_sim,
        "finetuned_model": ft_sim,
        "delta": ft_sim - base_sim
    })

# 打印结果
print("\n{:<50} {:<15} {:<15} {:<10}".format("Term Pair", "Base Model", "Finetuned", "Delta"))
for res in results:
    print("{:<50} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        f"{res['term_pair'][0]} vs {res['term_pair'][1]}",
        res['base_model'],
        res['finetuned_model'],
        res['delta']
    ))

# 可视化
labels = [f"{p[0]}\nvs\n{p[1]}" for p in term_pairs]
base_sims = [r['base_model'] for r in results]
ft_sims = [r['finetuned_model'] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, base_sims, width, label='Base Model')
rects2 = ax.bar(x + width/2, ft_sims, width, label='Finetuned Model')

ax.set_ylabel('Cosine Similarity')
ax.set_title('Embedding Similarity Comparison (Corrected)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

fig.tight_layout()
# plt.savefig("similarity_comparison_corrected.png", bbox_inches='tight', dpi=300)
plt.show()