---
layout: model
title: Japanese BERT Base
author: John Snow Labs
name: bert_base_japanese
date: 2021-09-16
tags: [bert, ja, open_source, japanese, embeddings]
task: Embeddings
language: ja
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture. It was originally published by

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2018.

The weights of this model are those released by the original BERT authors. The models are trained on the Japanese version of Wikipedia. The training corpus is generated from the Wikipedia Cirrussearch dump file as of August 31, 2020.

The generated corpus files are 4.0GB in total, consisting of approximately 30M sentences.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_japanese_ja_3.2.2_3.0_1631803687387.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_japanese", "ja") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_japanese", "ja")
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_japanese|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ja|
|Case sensitive:|true|

## Data Source

The models are trained on the Japanese version of Wikipedia. The training corpus is generated from the Wikipedia Cirrussearch dump file as of August 31, 2020.

The generated corpus files are 4.0GB in total, consisting of approximately 30M sentences.

https://github.com/cl-tohoku/bert-japanese