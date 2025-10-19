# KCFM: A Knowledge-Graph–Informed Foundation Model for Single-Cell Annotation**

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
For the step-by-step tutorial, please refer to: https://github.com/nudt-bioinfo/KCFM/tree/main/KCFM_tutorial

## Usage

### Fine-tune a PubMedBERT model
Cell Ontology (CO) is a standardized cell type classification system. As shown in Figure 1 (/reference/fig1.jpg), this system 
defines the connections between cell types, based on which we construct a hierarchical knowledge graph that formally represents 
relationships among cell types based on established cell ontologies (Fig. 1a). This graph captures ontological relations (Fig. 1c), 
encapsulating natural hierarchical and functional connections across cell types. We then fine-tune a PubMedBERT model using
these relational structures, generating type embeddings that are inherently enriched with ontological knowledge.

### Pretrained KCFM model

Single-cell sequencing has revolutionized understanding of cellular heterogeneity, but effectively interpreting its massive 
data and accurately annotating cell types remains challenging. Most existing methods only rely on gene expression profiles, 
overlooking structured biological knowledge of cell type relationships. To address this, we introduce KCFM, a novel multi-stage 
knowledge-guided framework that integrates prior biological knowledge into a foundational model for single-cell analysis.

KCFM model (e.g. **cell_cls_3loss_6layer_final.pth**) can be obtained.

### Downstream task
KCFM, a knowledge-guided foundational model designed to enhance single-cell data analysis by integrating structured biological 
knowledge from cell ontologies. Our approach constructs a hierarchical knowledge graph to formally represent cell type relationships, 
fine-tunes a PubMedBERT model to generate knowledge-aware cell type embeddings, and employs Mamba2-based modules with contrastive 
learning to align cell embeddings with ontological structures. This framework produces general-purpose cell representations that 
significantly improve performance across multiple downstream tasks—including fine-grained cell type annotation, novel cell discovery, 
spatial transcriptomics analysis under extreme data sparsity, and gene perturbation analysis—demonstrating both biological consistency 
and robustness to batch effects.

All downstream task codes are located in the **script** folder.

#### fine-grained cell type annotation
```angular2html
python ./T_cancer_cell/run_train_T_cancer_cell_classification.py
python ./T_cancer_cell/run_test_T_cancer_cell_classification.py
```

#### novel cell classification
```bash
python ./novel_cell_classification_bert/src/run_mamba_novel_cell_classification_difficulty.py
```

#### spatial transcriptomics analysis under extreme data sparsity
To provide a comprehensive evaluation of KCFM’s performance, we focused on a pervasive challenge in spatially resolved 
transcriptomics (ST): accurate cell-type annotation under data-scarce conditions. we use four carefully designed existing 
datasets (CL-intra, CL-cross, SB-intra, SB-cross). These datasets cover both intra-tissue and cross-tissue annotation scenarios, 
with each containing only approximately 50 genes, simulating data-scarce conditions commonly encountered in real studies.

**CL_cross**

Fine-tune the KCFM model using the CL_cross dataset, and then test the performance of the fine-tuned model on the test set.

```angular2html
python ./spatial_transcriptomics/run_train_CL_cross.py
python ./spatial_transcriptomics/run_test.py
```

**CL_intra**

Fine-tune the KCFM model using the CL_intra dataset, and then test the performance of the fine-tuned model on the test set.

```angular2html
python ./spatial_transcriptomics/run_train_CL_intra.py
python ./spatial_transcriptomics/run_test.py
```

**SB_cross**

Fine-tune the KCFM model using the SB_cross dataset, and then test the performance of the fine-tuned model on the test set.

```angular2html
python ./spatial_transcriptomics/run_train_SB_cross.py
python ./spatial_transcriptomics/run_test.py
```

**SB_intra**

Fine-tune the KCFM model using the SB_intra dataset, and then test the performance of the fine-tuned model on the test set.

```angular2html
python ./spatial_transcriptomics/run_train_SB_intra.py
python ./spatial_transcriptomics/run_test.py
```

**Cell Cycle Analysis**

Unlike CL indra, CL cross, SB indra, SB cross, the data of cell cycle analysis reflects the characteristic of extremely 
small sample size, that is, the sample size is scarce.
```angular2html
python ./cell_cycle_classification/run_cell_cycle_classification.py
```

#### gene perturbation analysis
```angular2html
python ./gene_pretubation/GEARS/gears/train.py
```

