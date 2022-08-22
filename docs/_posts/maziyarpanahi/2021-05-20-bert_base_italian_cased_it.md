---
layout: model
title: Italian BERT Base Cased
author: John Snow Labs
name: bert_base_italian_cased
date: 2021-05-20
tags: [open_source, it, italian, embeddings, bert]
task: Embeddings
language: it
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The source data for the Italian BERT model consists of a recent Wikipedia dump and various texts from the [OPUS corpora](http://opus.nlpl.eu/) collection. The final training corpus has a size of 13GB and 2,050,057,573 tokens.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_italian_cased_it_3.1.0_2.4_1621508025859.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_italian_cased", "it") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_italian_cased", "it")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("it.embed.bert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_italian_cased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|it|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/dbmdz/bert-base-italian-cased](https://huggingface.co/dbmdz/bert-base-italian-cased)