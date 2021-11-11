---
layout: model
title: Greek BERT Base Uncased Embedding
author: John Snow Labs
name: bert_base_uncased
date: 2021-09-07
tags: [greek, open_source, bert_embeddings, uncased, el]
task: Embeddings
language: el
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A Greek version of BERT pre-trained language model. 

The pre-training corpora of bert-base-greek-uncased-v1 include:

- The Greek part of Wikipedia,
- The Greek part of European Parliament Proceedings Parallel Corpus, and
- The Greek part of OSCAR, a cleansed version of Common Crawl.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_el_3.2.2_3.0_1630999695036.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_uncased", "el") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_uncased", "el")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|el|
|Case sensitive:|true|

## Data Source

The model is imported from: https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1