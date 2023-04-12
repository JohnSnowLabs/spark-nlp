---
layout: model
title: Chinese BertForTokenClassification Cased model (from canIjoin)
author: John Snow Labs
name: bert_token_classifier_datafun
date: 2023-03-22
tags: [zh, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: zh
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `datafun` is a Chinese model originally trained by `canIjoin`.

## Predicted Entities

`movie`, `no1`, `government`, `name1`, `position`, `book1`, `address`, `address1`, `game`, `organization`, `book`, `government1`, `company1`, `game1`, `position1`, `movie1`, `scene1`, `name`, `company`, `scene`, `organization1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_datafun_zh_4.3.1_3.0_1679492680049.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_datafun_zh_4.3.1_3.0_1679492680049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_datafun","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_datafun","zh") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_datafun|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|381.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/canIjoin/datafun
- https://github.com/dbiir/UER-py/wiki/Modelzoo
- https://github.com/CLUEbenchmark/CLUENER2020
- https://github.com/dbiir/UER-py/
- https://cloud.tencent.com/