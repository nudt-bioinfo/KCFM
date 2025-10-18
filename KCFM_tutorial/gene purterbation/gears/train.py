# -*- coding: utf-8 -*-
import argparse
import os
import torch

# 强烈建议把包内导入写成绝对导入
from gears.pertdata import PertData
from gears.gears import GEARS

ALLOWED_NAMES = {"norman", "adamson", "dixit"}

def main():
    p = argparse.ArgumentParser()
    # 必填：数据根目录（需包含 gene2go.pkl）
    p.add_argument("--data_root", type=str, default="/home/mjin/scFoundation-main/GEARS/data",
                   help="数据根目录，必须包含 gene2go.pkl，如 ./data")
    # 二选一：数据名（仅支持 norman/adamson/dixit 自动下载）或本地数据集目录/文件
    p.add_argument("--dataset_name", type=str, default=None,
                   help="可选，仅支持 {norman, adamson, dixit} 自动下载")
    p.add_argument("--dataset_path", type=str, default="/home/mjin/scFoundation-main/GEARS/data/",
                   help="本地加载：指向目录(含 perturb_processed.h5ad 或 norman.h5ad) 或直接指向 .h5ad 文件")
    # 划分与loader
    p.add_argument("--split", type=str, default="simulation")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--train_gene_set_size", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    # 训练超参
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--accumulation_steps", type=int, default=5)
    p.add_argument("--highres", type=int, default=0)
    p.add_argument("--singlecell_model_path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--result_dir", type=str, default="/home/mjin/scFoundation-main/GEARS/result/scfoundation-adamson")

    args = p.parse_args()

    # 0) 基本检查
    gene2go_pkl = os.path.join(args.data_root, "gene2go.pkl")
    if not os.path.exists(gene2go_pkl):
        raise FileNotFoundError(f"缺少 {gene2go_pkl}，PertData.__init__ 会读取它。请放到 data_root 下。")

    # 1) 构造 PertData（此处传 data_root）
    # gi_go=False 仅表示不基于 GI/GO 过滤基因；不影响 GEARS 后续是否尝试构建 GO 图
    pert_data = PertData(data_path=args.data_root, gi_go=False)

    # 2) 载入数据
    if args.dataset_name is not None:
        name = args.dataset_name.lower()
        if name not in ALLOWED_NAMES:
            raise ValueError(f"--dataset_name 仅支持 {ALLOWED_NAMES}，否则请改用 --dataset_path")
        pert_data.load(data_name=name)  # 自动下载到 data_root/name 下并生成 perturb_processed.h5ad
        print(f"[INFO] Using auto-downloaded dataset: {name}")
    else:
        if args.dataset_path is None:
            raise ValueError("必须提供 --dataset_name 或 --dataset_path 其一")
        if not os.path.exists(os.path.join(args.dataset_path, "adamson.h5ad")): raise FileNotFoundError(f"{args.dataset_path}/adamson.h5ad 不存在，请确认路径")
        # 允许传目录或 .h5ad 文件；若只有 norman.h5ad 则自动生成 perturb_processed.h5ad
        pert_data.load(data_path=args.dataset_path)
    # 3) 划分与 DataLoader
    pert_data.prepare_split(split=args.split,
                            seed=args.seed,
                            train_gene_set_size=args.train_gene_set_size)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size)

    # 4) GEARS 实例化与初始化
    gears_model = GEARS(pert_data, device=args.device)
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        model_type='emb',
        load_path=args.singlecell_model_path,
        finetune_method='random',
        accumulation_steps=args.accumulation_steps,
        highres=args.highres
    )
    # 5) 训练
    os.makedirs(args.result_dir, exist_ok=True)
    gears_model.train(epochs=args.epochs,
                      result_dir=args.result_dir,
                      lr=args.lr,
                      weight_decay=args.weight_decay)

if __name__ == "__main__":
    main()
