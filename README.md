# KCFM: A Knowledge-Graph–Informed Foundation Model for Single-Cell Annotation

## Overview
KCFM is a novel framework that integrates biological knowledge from cell ontologies with single-cell sequencing data to improve cell type classification. By combining gene expression profiles with structured biological knowledge, KCFM achieves state-of-the-art performance across multiple analysis tasks.
![KCFM framework](./workflow.png)

## Quick Start
```bash
conda create -n kcfm python=3.10
conda activate kcfm
pip install -r requirements.txt
```

## Tutorial

### Knowledge-Enhanced PubMedBERT Fine-tuning
We leverage the Cell Ontology (CO) hierarchy to construct a structured knowledge graph (Fig. 1a), capturing ontological relationships between cell types (Fig. 1c). This graph informs our fine-tuning of PubMedBERT, producing cell type embeddings that intrinsically encode biological relationships.

### Pretrained KCFM Model
Our pretrained model (`cell_cls_3loss_6layer_final.pth`) integrates:
- Biological knowledge from cell ontologies
- Gene expression patterns
- Contrastive learning objectives
- Mamba2-based architecture for efficient processing

### Benchmark Evaluation
We evaluated KCFM across four key scenarios:

1. **fine-grained cell type annotation**
```bash
python ./T_cancer_cell/run_train_T_cancer_cell_classification.py
python ./T_cancer_cell/run_test_T_cancer_cell_classification.py
```

2. **novel cell classification**
```bash
python ./novel_cell_classification_bert/src/run_mamba_novel_cell_classification_difficulty.py
```

3. **spatial transcriptomics analysis under extreme data sparsity**
Tested under four challenging conditions:

**Cross_tissue analysis**
```angular2html
python ./spatial_transcriptomics/run_train_{CL|SB}_cross.py
python ./spatial_transcriptomics/run_test.py
```

**​Intra-tissue analysis**
```angular2html
python ./spatial_transcriptomics/run_train_{CL|SB}_intra.py
python ./spatial_transcriptomics/run_test.py
```

4. **gene perturbation analysis**

```angular2html
python ./gene_pretubation/GEARS/gears/train.py
```

