# Release notes (data & path conventions)

- Tutorial notebooks use paths relative to the repo root (`../../` from
  `KCFM_tutorial/<task>/`). Unpack `data/*.zip` under `data/`; the cell-atlas
  dataset (link in README.md) should also be placed under
  `data/merlin_cxg_2023_05_15_sf-log1p/`.
- Perturbation tutorial (`KCFM_tutorial/gene purterbation/`): the per-gene
  embedding `.npy` is not distributed (size). Generate it from the released
  checkpoint, then point to it via `train.py --embedding_path <path>` or env
  `KCFM_GENE_EMB_PATH` (default `embeddings/norman_embeddings_2loss.npy`
  relative to the run directory).
- Evaluation outputs directory: env `GEARS_EVAL_OUTDIR` (default
  `results/eval_outputs`).
- Pretraining (`KCFM_tutorial/pretraining/train_pretrain.py`) implements the
  three-loss objective with paper defaults alpha=1.0 (MLM), beta=0.5 (CLS),
  gamma=1.0 (rank); 15% of non-padding positions masked, decoder target = the
  original expression values at masked positions; no tissue reweighting by
  default (`--tissue_weighting` reproduces the ablation). Run from repo root or
  pass `--data_root` / `--bert_dir` pointing at the unpacked datasets.
- Protocol unchanged: GEARS `simulation` split (seed 3, 75% training genes);
  each perturbed cell is paired with one randomly sampled control cell
  (`num_samples=1`).
