# -*- coding: utf-8 -*-
import argparse
import os
import torch

from gears.pertdata import PertData
from gears.gears import GEARS

ALLOWED_NAMES = {"norman", "adamson", "dixit"}

def main():
    p = argparse.ArgumentParser()
    # required: data root (must contain gene2go.pkl)
    p.add_argument("--data_root", type=str, default=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), "data"),
                   help="data root directory; must contain gene2go.pkl, e.g. ./data")
    # choose one: dataset name (norman/adamson/dixit auto-download) or local dataset path
    p.add_argument("--dataset_name", type=str, default=None,
                   help="optional; auto-download supports only {norman, adamson, dixit}")
    p.add_argument("--dataset_path", type=str, default=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), "data"),
                   help="local load: a directory (containing perturb_processed.h5ad or norman.h5ad) or a .h5ad file")
    # split and loaders
    p.add_argument("--split", type=str, default="simulation")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--train_gene_set_size", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    # training hyper-parameters
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--accumulation_steps", type=int, default=5)
    p.add_argument("--highres", type=int, default=0)
    p.add_argument("--singlecell_model_path", type=str, default=None)
    p.add_argument("--embedding_path", type=str, default=None,
                   help="per-gene embedding .npy; overrides KCFM_GENE_EMB_PATH env var")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--result_dir", type=str, default=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), "result"))

    args = p.parse_args()
    if args.embedding_path:
        os.environ["KCFM_GENE_EMB_PATH"] = args.embedding_path

    # 0) basic checks
    gene2go_pkl = os.path.join(args.data_root, "gene2go.pkl")
    if not os.path.exists(gene2go_pkl):
        raise FileNotFoundError(f"missing {gene2go_pkl}; PertData.__init__ reads it, please place it under data_root.")
    


    pert_data = PertData(data_path=args.data_root, gi_go=False)
    if args.dataset_name is not None:
        name = args.dataset_name.lower()
        if name not in ALLOWED_NAMES:
            raise ValueError(f"--dataset_name supports only {ALLOWED_NAMES}; otherwise use --dataset_path")
        pert_data.load(data_name=name)  
        print(f"[INFO] Using auto-downloaded dataset: {name}")
    else:
        if args.dataset_path is None:
            raise ValueError("provide either --dataset_name or --dataset_path")
        if not os.path.exists(os.path.join(args.dataset_path, "norman.h5ad")): raise FileNotFoundError(f"{args.dataset_path}/norman.h5ad not found; check the path")
        pert_data.load(data_path=args.dataset_path)
    # 3) split and DataLoader
    pert_data.prepare_split(split=args.split,
                            seed=args.seed,
                            train_gene_set_size=args.train_gene_set_size)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size)
    




    # 4) GEARS instantiation and init
    gears_model = GEARS(pert_data, device=args.device)
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        model_type='emb',
        load_path=args.singlecell_model_path,
        finetune_method='random',
        accumulation_steps=args.accumulation_steps,
        highres=args.highres
    )
    # 5) train
    os.makedirs(args.result_dir, exist_ok=True)
    gears_model.train(epochs=args.epochs,
                      result_dir=args.result_dir,
                      lr=args.lr,
                      weight_decay=args.weight_decay)

if __name__ == "__main__":
    main()
