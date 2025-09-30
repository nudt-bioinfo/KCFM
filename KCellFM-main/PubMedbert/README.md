---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
language: en
license: apache-2.0
---

# PubMedBERT Embeddings

This is a [PubMedBERT-base](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) model fined-tuned using [sentence-transformers](https://www.SBERT.net). It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search. The training dataset was generated using a random sample of [PubMed](https://pubmed.ncbi.nlm.nih.gov/) title-abstract pairs along with similar title pairs.

PubMedBERT Embeddings produces higher quality embeddings than generalized models for medical literature. Further fine-tuning for a medical subdomain will result in even better performance.

## Usage (txtai)

This model can be used to build embeddings databases with [txtai](https://github.com/neuml/txtai) for semantic search and/or as a knowledge source for retrieval augmented generation (RAG).

```python
import txtai

embeddings = txtai.Embeddings(path="neuml/pubmedbert-base-embeddings", content=True)
embeddings.index(documents())

# Run a query
embeddings.search("query to run")
```

## Usage (Sentence-Transformers)

Alternatively, the model can be loaded with [sentence-transformers](https://www.SBERT.net).

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
embeddings = model.encode(sentences)
print(embeddings)
```

## Usage (Hugging Face Transformers)

The model can also be used directly with Transformers. 

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")

# Tokenize sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    output = model(**inputs)

# Perform pooling. In this case, mean pooling.
embeddings = meanpooling(output, inputs['attention_mask'])

print("Sentence embeddings:")
print(embeddings)
```

## Evaluation Results

Performance of this model compared to the top base models on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is shown below. A popular smaller model was also evaluated along with the most downloaded PubMed similarity model on the Hugging Face Hub.

The following datasets were used to evaluate model performance.

- [PubMed QA](https://huggingface.co/datasets/pubmed_qa)
  - Subset: pqa_labeled, Split: train, Pair: (question, long_answer)
- [PubMed Subset](https://huggingface.co/datasets/zxvix/pubmed_subset_new)
  - Split: test, Pair: (title, text)
- [PubMed Summary](https://huggingface.co/datasets/scientific_papers)
  - Subset: pubmed, Split: validation, Pair: (article, abstract)

Evaluation results are shown below. The [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is used as the evaluation metric.

| Model                                                                         | PubMed QA | PubMed Subset | PubMed Summary | Average   |
| ----------------------------------------------------------------------------- | --------- | ------------- | -------------- | --------- | 
| [all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2)      | 90.40     | 95.86         | 94.07          | 93.44     | 
| [bge-base-en-v1.5](https://hf.co/BAAI/bge-large-en-v1.5)                      | 91.02     | 95.60         | 94.49          | 93.70     | 
| [gte-base](https://hf.co/thenlper/gte-base)                                   | 92.97     | 96.83         | 96.24          | 95.35     | 
| [**pubmedbert-base-embeddings**](https://hf.co/neuml/pubmedbert-base-embeddings) | **93.27** | **97.07**     | **96.58**      | **95.64** |
| [S-PubMedBert-MS-MARCO](https://hf.co/pritamdeka/S-PubMedBert-MS-MARCO)       | 90.86     | 93.33         | 93.54          | 92.58     | 

## Training

The model was trained with the parameters:

**DataLoader**:

`torch.utils.data.dataloader.DataLoader` of length 20191 with parameters:
```
{'batch_size': 24, 'sampler': 'torch.utils.data.sampler.RandomSampler', 'batch_sampler': 'torch.utils.data.sampler.BatchSampler'}
```

**Loss**:

`sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss` with parameters:
  ```
  {'scale': 20.0, 'similarity_fct': 'cos_sim'}
  ```

Parameters of the fit() method:
```
{
    "epochs": 1,
    "evaluation_steps": 500,
    "evaluator": "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.EmbeddingSimilarityEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'torch.optim.adamw.AdamW'>",
    "optimizer_params": {
        "lr": 2e-05
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 10000,
    "weight_decay": 0.01
}
```

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## More Information

Read more about this model and how it was built in [this article](https://medium.com/neuml/embeddings-for-medical-literature-74dae6abf5e0).
