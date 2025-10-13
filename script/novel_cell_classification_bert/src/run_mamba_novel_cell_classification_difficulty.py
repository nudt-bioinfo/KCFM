import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from transformers import AutoModel, AutoTokenizer, BertModel, BertConfig
from pathlib import Path

# 导入Mamba相关模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from models.model import MambaModel
from models.gene_tokenizer import GeneVocab


# 配置参数
class Config:
    # 数据路径
    pretrain_subset_path = "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/novel_cell_clssification/train_data_subset_0_01.h5ad"
    var_parquet_path = "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/var.parquet"
    cell_type_parquet_path = "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/categorical_lookup/cell_type.parquet"
    converted_data_dir = Path(
        "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/novel_cell_clssification/scCello_ood_celltype_data2/filtered_data_10_percent")
    json_relationship_path = Path("../data/celltype_relationship.json")

    triples_ontology_path = "../data/triples.txt"

    # 输出路径
    ontology_graph_path = "../data/cell_ontology_graph_scCello_data2.json"
    id_to_node_path = "../data/cell_id_to_node_id_mapping_scCello_data2.json"
    id_to_name_path = "../data/cell_id_to_cell_name_mapping_scCello_data2.json"
    cell_type_repr_path = "../data/cell_type_representations_mamba_scCello_data2.json"
    novel_cell_emb_path = "/mnt/HHD16T/DATA/lxz/sctab/merlin_cxg_2023_05_15_sf-log1p/novel_cell_clssification/data_hard_disk/mamba_bert/novel_cell_embeddings_mamba_bert_scCello_data2.pkl"
    cell_type_label_path = "../data/novel_cell_name_to_label_mamba_bert_scCello_data2.json"
    results_output_path = "../results/novel_cell_classification_results_mamba_bert_scCello_data2.csv"

    # Mamba模型配置 - 使用Mamba自身词表
    pretrained_model_path = "/home/lxz/scmamba/model_state/cell_cls_3loss_6layer_final.pth"
    gene_vocab_path = "/home/lxz/scmamba/vocab.json"  # Mamba自己的词表
    ensembl_ID_to_gene_name_path = "/home/lxz/scmamba/novel_cell_classification/data/ensembl_ID_to_gene_name_dict_gc30M.pkl"

    vocab = GeneVocab.from_file(gene_vocab_path)

    # bert模型配置
    pretrained_bert_model_path = "/home/lxz/PubMedbert"
    pretrained_bert_checkpoint_path = "/home/lxz/PubMedbert/finetune_bert.pth"

    # Mamba模型参数
    max_seq_len = 4096  # 序列最大长度
    ntoken = len(vocab)  # 词汇表大小，后续从词表加载
    embsize = 512  # 嵌入维度
    nhead = 8  # 注意力头数，Mamba中可能不使用但需保留
    d_hid = 512  # 隐藏层维度，Mamba中可能不使用但需保留
    nlayers = 6  # Mamba层数
    dropout = 0.1  # dropout率
    pad_token = "<pad>"  # pad标记
    pad_value = -2  # pad值
    input_emb_style = "continuous"  # 输入嵌入样式
    cell_emb_style = "cls"  # 细胞嵌入样式
    class_num = 164  # 分类数量

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_workers = 8

    # PPR参数
    alpha = 0.9
    threshold = 1e-4

    # 难度分级参数
    difficulty_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    selected_ratios = difficulty_ratios  # 默认运行所有难度等级
    num_samplings = 20  # 每个比例的采样次数: 20


# 1. 提取triples.txt中所有细胞类型名
"""从triplet.txt中提取所有is_a关系的细胞类型"""


def extract_is_a_cells(triplet_path):
    """从triplet.txt中提取所有is_a关系的细胞类型"""
    is_a_cells = set()  # 用集合自动去重
    try:
        with open(triplet_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去除首位空格
                if not line:  # 跳过空行
                    continue

                # 拆分包含空格的细胞名（通过" is_a "作为分隔符）
                if ' is_a ' in line:
                    cell1, cell2 = line.split(' is_a ', 1)
                    cell1 = cell1.strip()
                    cell2 = cell2.strip()

                    # 添加到集合（自动去重）
                    is_a_cells.add(cell1)
                    is_a_cells.add(cell2)
                else:
                    continue  # 跳过非is_a关系的行

        print(f"成功提取{len(is_a_cells)}种独特的is_a关系细胞类型")
        return is_a_cells

    except FileNotFoundError:
        print(f"错误：未找到triplet.txt文件（路径：{triplet_path}）")
        return set()
    except Exception as e:
        print(f"处理triplet.txt时出错：{str(e)}")
        return set()


# 2. Mamba模型包装类：用于提取细胞嵌入
class MambaEmbeddingExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_token_id = config.vocab[config.pad_token]

        # 加载基因词汇表（Mamba自身词表）
        self.vocab = GeneVocab.from_file(config.gene_vocab_path)
        config.ntoken = len(self.vocab)  # 动态设置词汇表大小

        # 初始化Mamba模型
        self.model = MambaModel(
            ntoken=config.ntoken,
            embsize=config.embsize,
            nhead=config.nhead,
            d_hid=config.d_hid,
            nlayers=config.nlayers,
            dropout=config.dropout,
            pad_token_id=self.pad_token_id,
            input_emb_style=config.input_emb_style,
            cell_emb_style=config.cell_emb_style
        )

        # 加载预训练权重
        self._load_pretrained_weights(config.pretrained_model_path, config.device)

        # 不微调模型，冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.pad_token_id = self.vocab[self.vocab.pad_token] if self.vocab.pad_token else 0
        self.cls_token_id = self.vocab["<cls>"] if "<cls>" in self.vocab else 1
        self.max_seq_len = config.max_seq_len

    def _load_pretrained_weights(self, model_path, device):
        try:
            pretrained_dict = torch.load(model_path, map_location=device)
            model_dict = self.model.state_dict()

            # 过滤不匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print("成功加载Mamba预训练权重")
        except Exception as e:
            print(f"加载Mamba预训练权重失败: {str(e)}")
            raise

    def forward(self, input_ids, values, attention_mask=None):
        # Mamba模型需要表达值作为输入
        src_key_padding_mask = (input_ids == self.pad_token_id) if attention_mask is None else attention_mask.bool()
        output = self.model(src=input_ids, values=values, src_key_padding_mask=src_key_padding_mask)
        return output["cell_emb"]


# 3. BERT微调模型
class finetune_BERT(nn.Module):
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


# 4. BERT嵌入提取器
class BertEmbeddingExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_model_path)
        self.finetuned_model = finetune_BERT(
            AutoModel.from_pretrained(config.pretrained_bert_model_path)
        ).to(config.device)

        # 加载预训练权重并设置为评估模式
        self.checkpoint = torch.load(config.pretrained_bert_checkpoint_path, map_location=config.device)
        self.finetuned_model.bert.load_state_dict(self.checkpoint['bert_state'])
        self.finetuned_model.projection.load_state_dict(self.checkpoint['projection_state'])
        self.finetuned_model.eval()  # 设置为评估模式

    def get_embedding(self, text):
        model = self.finetuned_model
        tokenizer = self.tokenizer
        device = self.config.device

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=25).to(device)
        with torch.no_grad():
            output = model(**inputs)
        return output.cpu().numpy()

# 新细胞数据集（适配Mamba模型）
class NovelCellDataset(Dataset):
    def __init__(self, parquet_files, ensembl_to_gene_name_dict):
        self.parquet_files = parquet_files
        self.ensembl_to_gene_name_dict = ensembl_to_gene_name_dict
        self.data = []

        # 加载Mamba自身词汇表
        self.vocab = GeneVocab.from_file(Config.gene_vocab_path)
        self.pad_token_id = self.vocab[self.vocab.pad_token] if self.vocab.pad_token else 0
        self.cls_token_id = self.vocab["<cls>"] if "<cls>" in self.vocab else 1

        # 获取模型最大序列长度
        self.max_seq_length = Config.max_seq_len

        # 预加载所有数据
        self._load_data()

    def _load_data(self):
        """加载所有parquet文件数据，过滤零表达值"""
        for file in tqdm(self.parquet_files, desc="加载新细胞数据"):
            try:
                df = pd.read_parquet(file)
                for _, row in df.iterrows():
                    # 处理非法数据：删除第一个元素
                    expr_nums = row['gene_expression_nums'][1:]
                    gene_names = row['gene_ensembl_ids'][1:]  # 基因名列表

                    # 转换为numpy数组便于过滤（参考非零值处理逻辑）
                    expr_array = np.array(expr_nums)
                    gene_array = np.array(gene_names, dtype=object)  # 保留None值的object类型

                    # 获取非零表达值的掩码（>0），参考提供的nonzero_mask逻辑
                    nonzero_mask = expr_array > 0
                    # 过滤得到非零表达值及其对应的基因名
                    nonzero_expr = expr_array[nonzero_mask]
                    nonzero_genes = gene_array[nonzero_mask]

                    # 映射token（仅处理非零表达的基因）
                    valid_tokens = []
                    valid_expr = []
                    for gene, expr in zip(nonzero_genes, nonzero_expr):
                        if gene is not None:  # 跳过None值基因
                            if gene.startswith("ENSG"):
                                gene_name = self.ensembl_to_gene_name_dict.get(gene, None)
                                if gene_name is not None and gene_name in self.vocab:
                                    valid_tokens.append(self.vocab[gene_name])
                                    valid_expr.append(expr)
                            else:
                                gene_name = gene
                                if gene_name is not None and gene_name in self.vocab:
                                    valid_tokens.append(self.vocab[gene_name])
                                    valid_expr.append(expr)

                    if valid_tokens:  # 确保有有效数据才添加
                        # 添加CLS标记
                        unsorted_tokens = [self.cls_token_id] + list(valid_tokens)
                        unsorted_expr = [0.0] + list(valid_expr)

                        self.data.append({
                            "expr": unsorted_expr,
                            "tokens": unsorted_tokens,
                            "cell_type": row['cell_type']
                        })
            except Exception as e:
                print(f"加载文件 {file.name} 时出错: {str(e)}")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        expr = item["expr"]
        cell_type = item["cell_type"]

        # 截断或填充到最大长度
        if len(tokens) >= self.max_seq_length:
            input_ids = tokens[:self.max_seq_length]
            values = expr[:self.max_seq_length]
            attention_mask = [0] * self.max_seq_length  # 0表示有效（与Mamba的src_key_padding_mask一致）
        else:
            padding_len = self.max_seq_length - len(tokens)
            input_ids = tokens + [self.pad_token_id] * padding_len
            values = expr + [Config.pad_value] * padding_len
            attention_mask = [0] * len(tokens) + [1] * padding_len  # 1表示填充

        return {
            "input_ids": torch.tensor(input_ids),
            "values": torch.tensor(values, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask),
            "cell_type": cell_type
        }

# 6. 生成新细胞嵌入表示
"""为新细胞生成嵌入表示"""


def generate_novel_cell_embeddings():
    """为新细胞生成嵌入表示"""
    if os.path.exists(Config.novel_cell_emb_path) and os.path.exists(Config.cell_type_label_path):
        print("新细胞嵌入及标签映射已存在，直接加载...")
        with open(Config.novel_cell_emb_path, 'rb') as f:
            novel_cell_data = pickle.load(f)
        with open(Config.cell_type_label_path, 'r') as f:
            cell_type_to_label = json.load(f)
        return novel_cell_data, cell_type_to_label

    print("开始生成新细胞嵌入...")
    # 生成细胞类型到数字标签的映射
    cell_types = set()
    parquet_files = sorted(Config.converted_data_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"错误：在 {Config.converted_data_dir} 中未找到parquet文件")
        return None, None

    for file in tqdm(parquet_files, desc="收集新细胞类型"):
        try:
            df = pd.read_parquet(file)
            current_types = set(df['cell_type'].unique())
            cell_types.update(current_types)
        except Exception as e:
            print(f"处理文件 {file.name} 时出错: {str(e)}")
            continue

    # 排序并生成标签映射
    sorted_cell_types = sorted(cell_types)
    cell_type_to_label = {cell_type: idx for idx, cell_type in enumerate(sorted_cell_types)}

    # 保存标签映射
    with open(Config.cell_type_label_path, 'w') as f:
        json.dump(cell_type_to_label, f, indent=2)

    try:
        with open(Config.ensembl_ID_to_gene_name_path, 'rb') as f:
            ensembl_to_gene_name_dict = pickle.load(f)
    except Exception as e:
        print(f"读取{Config.ensembl_ID_to_gene_name_path}失败：{str(e)}")
        return None, None

    # 创建数据集和数据加载器
    dataset = NovelCellDataset(parquet_files, ensembl_to_gene_name_dict)
    print(f"新细胞数据集加载完成，共包含 {len(dataset)} 个有效细胞")

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )

    # 加载Mamba模型
    model = MambaEmbeddingExtractor(Config).to(Config.device)
    model.eval()

    # 提取嵌入
    novel_cell_data = []  # 存储 (embedding, cell_type, label)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取新细胞嵌入"):
            input_ids = batch["input_ids"].to(Config.device)
            values = batch["values"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            cell_types = batch["cell_type"]

            with autocast():
                embeddings = model(input_ids, values, attention_mask=attention_mask)

            embeddings_np = embeddings.cpu().numpy()

            # 保存嵌入和对应的细胞类型
            for emb, cell_type in zip(embeddings_np, cell_types):
                label = cell_type_to_label[cell_type]
                novel_cell_data.append((emb, cell_type, label))

    # 保存新细胞嵌入
    with open(Config.novel_cell_emb_path, 'wb') as f:
        pickle.dump(novel_cell_data, f)

    print(f"新细胞嵌入已保存至 {Config.novel_cell_emb_path}")
    return novel_cell_data, cell_type_to_label


# 7. 生成新细胞类型的BERT嵌入
def generate_novel_cell_type_embeddings():
    if os.path.exists(Config.cell_type_label_path):
        print("新细胞类型已存在，直接加载...")
        with open(Config.cell_type_label_path, 'r') as f:
            novel_cell_type_to_label = json.load(f)
    else:
        print("新细胞类型不存在，程序退出")
        exit()

    bertEmbeddingExtractor = BertEmbeddingExtractor(Config)

    novel_cell_type_emb = []
    novel_cell_type_emb_dict = {}
    for cell_type in tqdm(novel_cell_type_to_label.keys(), desc="生成新细胞类型嵌入"):
        emb = bertEmbeddingExtractor.get_embedding(cell_type)
        novel_cell_type_emb.append(emb.squeeze(0))
        novel_cell_type_emb_dict[cell_type] = emb.squeeze(0)

    novel_cell_type_emb = np.array(novel_cell_type_emb)
    return novel_cell_type_emb_dict, novel_cell_type_emb


# 8. 生成知识图谱中细胞类型的BERT嵌入
def generate_ontology_cell_type_emb():
    if os.path.exists(Config.triples_ontology_path):
        print("知识图谱细胞类型已存在，直接加载...")
        ontology_cell_type_unique = extract_is_a_cells(Config.triples_ontology_path)
    else:
        print("知识图谱细胞类型不存在，程序退出")
        exit()

    bertEmbeddingExtractor = BertEmbeddingExtractor(Config)

    ontology_cell_type_emb = []
    ontology_cell_type_emb_dict = {}
    for cell_type in tqdm(ontology_cell_type_unique, desc="生成知识图谱中细胞类型的嵌入表示"):
        emb = bertEmbeddingExtractor.get_embedding(cell_type)
        ontology_cell_type_emb.append(emb.squeeze(0))
        ontology_cell_type_emb_dict[cell_type] = emb.squeeze(0)
    ontology_cell_type_emb = np.array(ontology_cell_type_emb)
    return ontology_cell_type_emb_dict, ontology_cell_type_emb


# 9. 细胞类型名称处理工具函数
"""将细胞类型名称转换为小写的映射字典"""


def get_upper_cell_name_to_lower_cell_name_mapping(cell_name_list):
    """将细胞类型名称转换为小写的映射字典"""
    cell_name_lower_dict = {}
    for name in cell_name_list:
        new_name = name.lower()
        cell_name_lower_dict[name] = new_name
    return cell_name_lower_dict


# 10. 对新细胞进行分类（新增返回`filtered_ontology_types`用于后续索引映射）
"""
    对新细胞进行分类：
    1. 仅考虑novel_cell_type_to_label中的细胞类型
    2. 计算每个细胞嵌入与这些类型嵌入的余弦相似度
    3. 找到最相似的类型作为预测
    4. 与真实类型比较，生成分类报告
    """


def classify_novel_cells(novel_cell_data, ontology_cell_type_emb_dict, novel_cell_type_to_label):
    """
    对新细胞进行分类：
    1. 仅考虑novel_cell_type_to_label中的细胞类型
    2. 计算每个细胞嵌入与这些类型嵌入的余弦相似度
    3. 找到最相似的类型作为预测
    4. 与真实类型比较，生成分类报告
    """
    # 准备数据
    cell_embeddings = np.array([item[0] for item in novel_cell_data])  # 细胞嵌入
    true_labels = np.array([item[2] for item in novel_cell_data])  # 真实标签
    true_cell_types = [item[1] for item in novel_cell_data]  # 真实细胞类型名称

    # 获取novel_cell_type_to_label中的细胞类型（小写形式）
    novel_types = set(novel_cell_type_to_label.keys())
    novel_types_lower = {t.lower() for t in novel_types}

    # 从本体论中筛选出存在于novel_cell_type_to_label中的类型
    filtered_ontology_types = []
    filtered_ontology_embs = []

    # 创建本体论类型到小写的映射
    all_ontology_types = list(ontology_cell_type_emb_dict.keys())
    ontology_type_lower = get_upper_cell_name_to_lower_cell_name_mapping(all_ontology_types)

    # 筛选过程
    for ont_type in all_ontology_types:
        ont_type_lower = ontology_type_lower[ont_type]
        if ont_type_lower in novel_types_lower:
            filtered_ontology_types.append(ont_type)
            filtered_ontology_embs.append(ontology_cell_type_emb_dict[ont_type])

    # 检查筛选结果
    print(f"从本体论中筛选出 {len(filtered_ontology_types)} 种存在于新细胞数据中的类型")
    if len(filtered_ontology_types) == 0:
        print("错误：未找到任何重叠的细胞类型，无法进行分类")
        return None, None, None, None, None

    # 转换为numpy数组
    filtered_ontology_embs = np.array(filtered_ontology_embs)

    # 创建筛选后的类型到小写的映射和标签映射
    filtered_type_lower = get_upper_cell_name_to_lower_cell_name_mapping(filtered_ontology_types)
    novel_type_lower = {k.lower(): v for k, v in novel_cell_type_to_label.items()}

    # 计算所有细胞与筛选后的类型嵌入的余弦相似度
    print("计算细胞嵌入与筛选后的类型嵌入的余弦相似度...")
    similarities = cosine_similarity(cell_embeddings, filtered_ontology_embs)  # 形状: [n_cells, n_filtered_types]

    ########################################
    # """使用spearmanr测量向量之间的相关性"""
    # # 将数据转换为秩次
    # cell_ranks = rankdata(cell_embeddings, axis=1)
    # ontology_ranks = rankdata(filtered_ontology_embs, axis=1)
    #
    # # 计算相关性（1 - 秩次距离）
    # # 注意：这里使用相关性而不是距离，所以用1减去
    # similarities = 1 - cdist(cell_ranks, ontology_ranks, metric='correlation')
    ########################################

    # 预测每个细胞的类型
    pred_labels = []
    for i in range(len(cell_embeddings)):
        # 找到最相似的筛选后类型（仅在新细胞类型范围内）
        max_sim_idx = np.argmax(similarities[i])
        pred_type = filtered_ontology_types[max_sim_idx]
        pred_type_lower = filtered_type_lower[pred_type]  # 转换为小写

        # 获取对应的标签（此时应该一定存在）
        pred_label = novel_type_lower[pred_type_lower]
        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels)

    # 新增返回：本体论重叠类型列表（用于后续索引映射）
    return true_labels, true_cell_types, pred_labels, similarities, filtered_ontology_types


# 11. 难度分级评估函数
"""
    按照不同难度等级评估分类性能，基于筛选后的类别重新计算预测
    selected_ratios: 可选，指定要评估的难度等级
    """


def evaluate_with_difficulty_levels(true_labels, true_cell_types, pred_labels, similarities,
                                    filtered_ontology_types, novel_cell_type_to_label, selected_ratios=None):
    """
    按照不同难度等级评估分类性能，基于筛选后的类别重新计算预测
    selected_ratios: 可选，指定要评估的难度等级
    """
    if selected_ratios is None:
        selected_ratios = Config.selected_ratios

    # 关键修复：建立「本体论重叠类型」与「新细胞类型」的映射
    # 1. 新细胞类型→小写的映射
    novel_type_lower = {k.lower(): k for k in novel_cell_type_to_label.keys()}
    # 2. 本体论重叠类型→新细胞类型的映射（通过小写匹配）
    filtered_to_novel_type = {}
    for ont_type in filtered_ontology_types:
        ont_type_lower = ont_type.lower()
        if ont_type_lower in novel_type_lower:
            filtered_to_novel_type[ont_type] = novel_type_lower[ont_type_lower]

    # 3. 本体论重叠类型列表（相似度矩阵的列对应顺序）
    n_filtered_types = len(filtered_ontology_types)
    all_result, avg_result = [], []

    # 创建标签到细胞类型的映射
    label_to_type = {v: k for k, v in novel_cell_type_to_label.items()}

    # 确保输出目录存在
    os.makedirs(os.path.dirname(Config.results_output_path), exist_ok=True)

    for ratio in selected_ratios:
        # 计算本次要选中的类型数（基于本体论重叠类型数，避免超界）
        num_selected = int(n_filtered_types * ratio + 0.5)
        num_selected = max(1, num_selected)  # 确保至少选择1种类型

        print(f"\n评估难度等级: {ratio} (选择 {num_selected}/{n_filtered_types} 种本体论重叠类型)")

        all_acc, all_f1 = [], []
        num_samplings = 1 if ratio >= 1.0 else Config.num_samplings

        for _ in range(num_samplings):
            # 修复1：基于「本体论重叠类型数」生成索引（确保不超界）
            selected_filtered_indices = np.random.permutation(n_filtered_types)[:num_selected]
            # 获取选中的本体论类型
            selected_filtered_types = [filtered_ontology_types[i] for i in selected_filtered_indices]
            # 映射到对应的新细胞类型
            selected_novel_types = [filtered_to_novel_type[t] for t in selected_filtered_types]
            # 获取新细胞类型对应的原始标签
            selected_novel_labels = [novel_cell_type_to_label[t] for t in selected_novel_types]

            # 创建新的标签映射（选中类型内重新编号）
            new_label_mapping = {old_label: idx for idx, old_label in enumerate(selected_novel_labels)}

            # 筛选出属于选中类型的细胞
            used_indices = np.isin(true_labels, selected_novel_labels)

            if np.sum(used_indices) == 0:
                print("警告：未找到选中类型的细胞，跳过此次采样")
                continue

            # 重新映射真实标签
            new_true_labels = [new_label_mapping[x] for x in np.array(true_labels)[used_indices]]

            # 修复2：用「本体论重叠类型索引」索引相似度矩阵列（确保不超界）
            selected_similarities = similarities[used_indices][:, selected_filtered_indices]
            new_preds = np.argmax(selected_similarities, axis=1)

            # 计算指标
            acc = accuracy_score(new_true_labels, new_preds)
            f1 = f1_score(new_true_labels, new_preds, average="macro")

            all_acc.append(acc)
            all_f1.append(f1)

            # 对于完整数据集，输出详细报告
            if ratio >= 1.0:
                print("\n完整分类报告:")
                print(classification_report(
                    new_true_labels,
                    new_preds,
                    target_names=selected_novel_types,
                    digits=4,
                    zero_division=0
                ))
                break

        print(f"难度等级：{ratio}")
        print(f"acc:")
        print(all_acc)
        print(f"f1:")
        print(all_f1)

        # 计算平均值
        divnum = len(all_acc) if len(all_acc) > 0 else 1
        avg_acc = round(sum(all_acc) / divnum, 4) if divnum > 0 else 0
        avg_f1 = round(sum(all_f1) / divnum, 4) if divnum > 0 else 0

        avg_result.append((ratio, avg_acc, avg_f1))
        all_result.append((ratio, all_acc, all_f1))

        print(f"难度等级 {ratio} - 平均准确率: {avg_acc}, 平均F1分数: {avg_f1}")

    # 保存结果
    avg_result_df = pd.DataFrame(avg_result, columns=["ratio", "acc", "f1"])
    avg_result_df.to_csv(Config.results_output_path, index=False)
    print(f"\n评估结果已保存至 {Config.results_output_path}")

    return avg_result_df


# 主函数
def main():
    # 步骤1: 提取知识图谱中的细胞类型
    triples_cell_type_unique = extract_is_a_cells(Config.triples_ontology_path)
    if not triples_cell_type_unique:
        print("无法继续执行后续任务，因为未提取到is_a关系的细胞类型")
        return

    # 步骤2: 生成新细胞嵌入
    novel_cell_data, cell_type_to_label = generate_novel_cell_embeddings()
    if novel_cell_data is None:
        print("无法生成新细胞嵌入，程序退出")
        return

    # 步骤3: 使用bert模型为新细胞类型生成嵌入
    print("生成新细胞类型的BERT嵌入...")
    _, _ = generate_novel_cell_type_embeddings()

    # 步骤4: 使用bert模型为知识图谱中的细胞类型生成嵌入
    print("生成知识图谱细胞类型的BERT嵌入...")
    ontology_cell_type_emb_dict, _ = generate_ontology_cell_type_emb()

    # 步骤5: 分类新细胞数据（新增接收filtered_ontology_types）
    print("开始分类新细胞...")
    true_labels, true_cell_types, pred_labels, similarities, filtered_ontology_types = classify_novel_cells(
        novel_cell_data, ontology_cell_type_emb_dict, cell_type_to_label)

    if true_labels is None:
        print("分类失败，无法进行难度分级评估")
        return

    # 步骤6: 按难度等级评估（新增传递filtered_ontology_types）
    evaluate_with_difficulty_levels(
        true_labels,
        true_cell_types,
        pred_labels,
        similarities,
        filtered_ontology_types,  # 新增参数：本体论重叠类型列表
        cell_type_to_label
    )


if __name__ == "__main__":
    main()