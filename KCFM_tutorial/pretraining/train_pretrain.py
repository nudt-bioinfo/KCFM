# -*- coding: utf-8 -*-
"""
KCFM pretraining (multi-task: masked expression modeling + classification +
hierarchical ranking), matching the Methods section of the manuscript.

Usage:
    python train_pretrain.py --data_root ../../data/merlin_cxg_2023_05_15_sf-log1p

Defaults implement the paper configuration:
    L_total = alpha*L_MLM + beta*L_CLS + gamma*L_rank,  alpha=1.0 beta=0.5 gamma=1.0
    15% of non-padding gene positions are masked per cell; the masked expression
    value in the input is replaced by a sentinel (-1.0) and the decoder target is
    the ORIGINAL value at exactly those masked positions.
    No tissue-level reweighting or balanced sampling is applied (as stated in the
    manuscript); pass --tissue_weighting to enable the inverse-sqrt-count variant.
"""
import argparse
import os
import random
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from models.gene_tokenizer import GeneVocab  # noqa: E402
from models.model import MambaModel  # noqa: E402


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "../.."))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=os.path.join(repo, "data", "merlin_cxg_2023_05_15_sf-log1p"),
                   help="dataset dir with train/ parquet parts, var.parquet and categorical_lookup/")
    p.add_argument("--vocab_path", default=os.path.join(repo, "models", "vocab.json"))
    p.add_argument("--bert_dir", default=os.path.join(repo, "PubMedbert"),
                   help="PubMedBERT dir incl. finetune_bert.pth and Knowledge-triples.txt")
    p.add_argument("--start_file", default="part.0.parquet")
    p.add_argument("--end_file", default="part.465.parquet")
    p.add_argument("--out_dir", default=os.path.join(here, "checkpoints"))
    p.add_argument("--init_from", default=None, help="optional checkpoint to warm-start from")
    # model / optimization (paper defaults)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--embsize", type=int, default=512)
    p.add_argument("--nlayers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=2049)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--pad_value", type=float, default=-2.0)
    p.add_argument("--mask_value", type=float, default=-1.0)
    p.add_argument("--alpha", type=float, default=1.0, help="weight of L_MLM (paper config)")
    p.add_argument("--beta", type=float, default=0.5, help="weight of L_CLS (paper config)")
    p.add_argument("--gamma", type=float, default=1.0, help="weight of L_rank (paper config)")
    p.add_argument("--margin1", type=float, default=0.1, help="self-vs-close margin")
    p.add_argument("--margin2", type=float, default=0.2, help="close-vs-far margin")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--tissue_weighting", action="store_true",
                   help="apply inverse-sqrt tissue-count weights to L_CLS (ablation only)")
    return p.parse_args()


def build_relation_graph(triplets_file):
    graph = defaultdict(lambda: {"close": set(), "far": set()})
    with open(triplets_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], " ".join(parts[2:])
            if r in ["is_a", "exact_synonyms", "develops_from", "develops_into"]:
                graph[h]["close"].add(t)
                graph[t]["close"].add(h)
            elif r == "disjoint_from":
                graph[h]["far"].add(t)
                graph[t]["far"].add(h)
    return graph


def generate_masked_positions(padding_mask, mask_prob=0.15, device="cuda"):
    mask = torch.zeros_like(padding_mask, dtype=torch.float32, device=device)
    for i in range(padding_mask.size(0)):
        non_padding = torch.where(~padding_mask[i])[0]
        n_mask = int(len(non_padding) * mask_prob)
        if n_mask > 0:
            masked_idx = torch.randperm(len(non_padding), device=device)[:n_mask]
            mask[i, non_padding[masked_idx]] = 1.0
    return mask.bool()


def masked_mse_loss(input, target, mask):
    """MSE between input and target restricted to masked positions."""
    mask = mask.float()
    return F.mse_loss(input * mask, target * mask, reduction="sum") / (mask.sum() + 1e-8)


def hierarchical_ranking_loss(cell_emb, cell_type_ids, type_embeddings, relation_graph,
                              all_cell_types, margin1=0.1, margin2=0.2):
    """Two-margin triplet ranking loss aligning cell embeddings with the
    Cell Ontology hierarchy (self > close > far)."""
    cell_emb = F.normalize(cell_emb, dim=1)
    type_embeddings = F.normalize(type_embeddings, dim=1)
    losses = []
    for i in range(cell_emb.size(0)):
        anchor = cell_emb[i]
        ct = all_cell_types[cell_type_ids[i].item()]
        rel = relation_graph.get(ct, {"close": set(), "far": set()})

        self_emb = type_embeddings[cell_type_ids[i]]
        self_sim = F.cosine_similarity(anchor, self_emb, dim=0)

        close_embs = [type_embeddings[all_cell_types.index(t)]
                      for t in rel["close"] if t in all_cell_types]
        if not close_embs:
            close_embs = [type_embeddings[random.choice(
                [j for j in range(len(all_cell_types)) if j != cell_type_ids[i]])]]
        close_embs = torch.stack(close_embs)
        close_sim = F.cosine_similarity(anchor.unsqueeze(0), close_embs)
        max_close_sim = close_sim.max()

        neg_candidates = [idx for idx, t in enumerate(all_cell_types)
                          if t != ct and t not in rel["close"]]
        if not neg_candidates:
            neg_candidates = [cell_type_ids[i].item()]
        neg_emb = type_embeddings[random.choice(neg_candidates)]

        losses.append(
            F.relu(max_close_sim - self_sim + margin1)
            + F.relu(F.cosine_similarity(anchor, neg_emb, dim=0) - max_close_sim + margin2)
        )
    return torch.stack(losses).mean()


class ParquetCellDataset(Dataset):
    """Streams parquet parts assigned to this rank (interleaved by index)."""

    def __init__(self, path, rank, world_size, vocab, feature_names,
                 start_file=None, end_file=None, max_seq_len=2049):
        self.path = path
        self.rank = rank
        self.world_size = world_size
        self.vocab = vocab
        self.feature_names = feature_names
        self.max_seq_len = max_seq_len

        all_files = sorted([f for f in os.listdir(path) if f.endswith(".parquet")],
                           key=lambda x: int(x.split(".")[1]))
        if start_file or end_file:
            start_idx = 0
            end_idx = len(all_files)
            if start_file:
                start_idx = next((i for i, f in enumerate(all_files) if f == start_file), 0)
            if end_file:
                end_idx = next((i for i, f in enumerate(all_files) if f == end_file), len(all_files)) + 1
            all_files = all_files[start_idx:end_idx]
        self.assigned_files = [os.path.join(path, f) for i, f in enumerate(all_files)
                               if i % world_size == rank]
        print(f"Rank {rank} assigned {len(self.assigned_files)} files.")
        self.file_idx = 0
        self.df = None
        self.row_idx = 0
        self._load()

    def _load(self):
        if self.file_idx >= len(self.assigned_files):
            self.df = None
            return
        print(f"Rank {self.rank} loading {self.assigned_files[self.file_idx]}")
        self.df = pq.read_table(self.assigned_files[self.file_idx]).to_pandas()
        self.row_idx = 0

    def __len__(self):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.assigned_files)

    def construct_cell_sequence(self, expr):
        device = "cpu"
        expr = torch.as_tensor(expr, dtype=torch.float32, device=device)
        nonzero_idx = torch.nonzero(expr).squeeze()
        nonzero_vals = expr[nonzero_idx]
        max_seq_len = self.max_seq_len

        if len(nonzero_vals) < max_seq_len - 1:
            values = torch.cat([
                torch.tensor([0.0], device=device),
                nonzero_vals,
                torch.full((max_seq_len - 1 - len(nonzero_vals),), -2.0, device=device),
            ])
            extended_idx = torch.cat([
                torch.tensor([-1], device=device),
                nonzero_idx,
                torch.full((max_seq_len - 1 - len(nonzero_idx),), -1, device=device),
            ])
        else:
            perm = torch.randperm(len(nonzero_idx))[:max_seq_len - 1]
            extended_idx = nonzero_idx[perm]
            values = torch.cat([torch.tensor([0.0], device=device), expr[extended_idx]])

        valid_idx = extended_idx[extended_idx >= 0]
        genes = ["<cls>"] + [self.feature_names[i] for i in valid_idx.cpu().numpy()]
        genes += ["<pad>"] * (max_seq_len - len(genes))
        padding_mask = torch.tensor([g == "<pad>" for g in genes], dtype=torch.bool)
        src = torch.tensor([self.vocab[g] if g in self.vocab else self.vocab["<unk>"]
                            for g in genes], dtype=torch.long)
        return values, src, padding_mask

    def __getitem__(self, _):
        while True:
            if self.df is None or self.row_idx >= len(self.df):
                self.file_idx += 1
                self._load()
                if self.df is None:
                    raise StopIteration
                continue
            row = self.df.iloc[self.row_idx]
            self.row_idx += 1
            values, src, padding_mask = self.construct_cell_sequence(row["X"])
            return {
                "values": values,
                "src": src,
                "padding_mask": padding_mask,
                "cell_type": row["cell_type"],
                "tissue": row["tissue"],
            }


def make_collate(tissue_weighting):
    def collate(batch):
        values_list = [b["values"] for b in batch]
        src_list = [b["src"] for b in batch]
        padding_mask_list = [b["padding_mask"] for b in batch]
        cell_types_list = [b["cell_type"] for b in batch]
        out = {
            "values": torch.stack(values_list),
            "src": torch.stack(src_list),
            "padding_mask": torch.stack(padding_mask_list),
            "cell_type": torch.tensor(cell_types_list, dtype=torch.long),
        }
        if tissue_weighting:
            tissues_list = [b["tissue"] for b in batch]
            counts = pd.DataFrame({"tissue": tissues_list})["tissue"].value_counts()
            w = torch.tensor([1.0 / (counts[t] ** 0.5) for t in tissues_list],
                             dtype=torch.float32)
            w = w / w.sum() * len(w)  # normalize to keep gradient scale stable
            out["sample_weights"] = w
        return out
    return collate


class FinBERT(nn.Module):
    """Frozen PubMedBERT + projection to the expression embedding size."""

    def __init__(self, bert, d_model=512):
        super().__init__()
        self.bert = bert
        self.proj = nn.Sequential(nn.Linear(768, d_model), nn.ReLU(), nn.LayerNorm(d_model))

    def forward(self, **inputs):
        return self.proj(self.bert(**inputs).last_hidden_state.mean(1))


def setup_dist(rank, world_size, port):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_dist():
    dist.destroy_process_group()


def train(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    warnings.filterwarnings("ignore")

    vocab = GeneVocab.from_file(args.vocab_path)
    var = pd.read_parquet(os.path.join(args.data_root, "var.parquet"))
    feature_names = var["feature_name"].tolist()

    model = MambaModel(
        ntoken=len(vocab), d_model=args.embsize, nlayers=args.nlayers,
        vocab=vocab, dropout=args.dropout, pad_token="<pad>", pad_value=args.pad_value,
        input_emb_style="continuous", cell_emb_style="cls",
    ).to(device)
    if args.init_from:
        pretrained = torch.load(args.init_from, map_location=device)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

    setup_dist(rank, world_size, args.master_port)
    model = DDP(model, device_ids=[rank])
    scaler = torch.cuda.amp.GradScaler()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    bert_model = AutoModel.from_pretrained(args.bert_dir).to(device)
    finbert = FinBERT(bert_model, d_model=args.embsize).to(device)
    ckpt = torch.load(os.path.join(args.bert_dir, "finetune_bert.pth"), map_location=device)
    finbert.bert.load_state_dict(ckpt["bert_state"])
    finbert.proj.load_state_dict(ckpt["projection_state"])
    finbert.eval()

    df_ct = pd.read_parquet(os.path.join(args.data_root, "categorical_lookup", "cell_type.parquet"))
    all_cell_types = df_ct["label"].tolist()
    with torch.no_grad():
        ct_embs = [finbert(**tokenizer(ct, return_tensors="pt", truncation=True,
                                       max_length=25).to(device)).cpu().numpy()
                   for ct in tqdm(all_cell_types, desc="Cell-type embeddings", disable=rank != 0)]
    celltype_emb_tensor = torch.tensor(np.concatenate(ct_embs), device=device).detach()
    relation_graph = build_relation_graph(os.path.join(args.bert_dir, "Knowledge-triples.txt"))

    dataset = ParquetCellDataset(
        os.path.join(args.data_root, "train"), rank, world_size, vocab, feature_names,
        start_file=args.start_file, end_file=args.end_file, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=make_collate(args.tissue_weighting))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            try:
                values = batch["values"].to(device)
                src = batch["src"].to(device)
                pad_mask = batch["padding_mask"].to(device)
                cell_types = batch["cell_type"].to(device)

                mask_pos = generate_masked_positions(pad_mask, mask_prob=args.mask_ratio,
                                                     device=device)
                values_masked = values.clone()
                values_masked[mask_pos] = args.mask_value

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model(src=src, values=values_masked, src_key_padding_mask=pad_mask)
                    # MLM: reconstruct ORIGINAL expression values at masked positions
                    loss_mlm = masked_mse_loss(out["mlm_output"], values, mask_pos)
                    if args.tissue_weighting:
                        loss_cls = (criterion_cls(out["cls_output"], cell_types)
                                    * batch["sample_weights"].to(device)).mean()
                    else:
                        loss_cls = criterion_cls(out["cls_output"], cell_types).mean()
                    loss_rank = hierarchical_ranking_loss(
                        out["cell_emb"], cell_types, celltype_emb_tensor,
                        relation_graph, all_cell_types,
                        margin1=args.margin1, margin2=args.margin2)
                    total_loss = args.alpha * loss_mlm + args.beta * loss_cls + args.gamma * loss_rank

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Total {total_loss.item():.4f}, "
                          f"MLM {loss_mlm.item():.4f}, CLS {loss_cls.item():.4f}, "
                          f"Rank {loss_rank.item():.4f}")
                    if batch_idx % args.save_every == 0 and batch_idx != 0:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.out_dir, "kcfm_pretrained.pth"))
            except StopIteration:
                break
            except Exception as e:
                print(f"[ERROR] rank={rank} batch={batch_idx}: {e}")
                continue

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.out_dir, "kcfm_pretrained.pth"))
    cleanup_dist()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required (NCCL DDP).")
    args.master_port = int(os.environ.get("KCFM_MASTER_PORT", 12355))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
