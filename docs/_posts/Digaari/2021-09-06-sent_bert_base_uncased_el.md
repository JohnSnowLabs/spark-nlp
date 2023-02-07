---
layout: model
title: Greek BERT Sentence Base Uncased Embedding
author: John Snow Labs
name: sent_bert_base_uncased
date: 2021-09-06
tags: [greek, open_source, bert_sentence_embeddings, uncased, el]
task: Embeddings
language: el
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
annotator: BertSentenceEmbeddings
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_el_3.2.2_3.0_1630926274392.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_el_3.2.2_3.0_1630926274392.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased", "el") \
      .setInputCols("sentence") \
      .setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased", "el")
      .setInputCols("sentence")
      .setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_uncased|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|el|
|Case sensitive:|true|

## Data Source

The model is imported from: https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1