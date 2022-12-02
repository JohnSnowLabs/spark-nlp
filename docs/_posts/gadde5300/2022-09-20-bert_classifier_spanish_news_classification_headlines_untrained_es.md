---
layout: model
title: Spanish BertForSequenceClassification Cased model (from M47Labs)
author: John Snow Labs
name: bert_classifier_spanish_news_classification_headlines_untrained
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, es]
task: Text Classification
language: es
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `spanish_news_classification_headlines_untrained` is a Spanish model originally trained by `M47Labs`.

## Predicted Entities

`deportes`, `opinion`, `politica`, `medio_ambiente`, `educacion`, `cultura`, `ciencia_tecnologia`, `sociedad`, `clickbait`, `economia`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_spanish_news_classification_headlines_untrained_es_4.2.0_3.0_1663668733179.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_spanish_news_classification_headlines_untrained","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Amo Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_spanish_news_classification_headlines_untrained","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_spanish_news_classification_headlines_untrained|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/M47Labs/spanish_news_classification_headlines_untrained
- https://www.m47labs.com/es/
- https://colab.research.google.com/drive/1XsKea6oMyEckye2FePW_XN7Rf8v41Cw_?usp=sharing