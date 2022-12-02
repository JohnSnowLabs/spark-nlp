---
layout: model
title: BERT Biomedical Embeddings
author: John Snow Labs
name: bert_biomed_pubmed_uncased
date: 2022-02-21
tags: [pubmed, biomedical, bert, microsoft, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This embeddings model was imported from `Hugging Face` ([link](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)). It is pre trained from scratch using abstracts from PubMed and full-text articles from PubMedCentral. This model achieves state-of-the-art performance on many biomedical NLP tasks, and currently holds the top score on the Biomedical Language Understanding and Reasoning Benchmark.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_biomed_pubmed_uncased_en_3.4.0_3.0_1645440129466.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_biomed_pubmed_uncased", "en")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_biomed_pubmed_uncased", "en")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_biomed_pubmed_uncased|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|411.0 MB|
|Case sensitive:|true|

## References

[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)

```
@misc{pubmedbert,
  author = {Yu Gu and Robert Tinn and Hao Cheng and Michael Lucas and Naoto Usuyama and Xiaodong Liu and Tristan Naumann and Jianfeng Gao and Hoifung Poon},
  title = {Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing},
  year = {2020},
  eprint = {arXiv:2007.15779},
}
```
