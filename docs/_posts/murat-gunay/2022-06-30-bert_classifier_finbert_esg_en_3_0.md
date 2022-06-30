---
layout: model
title: English ESG BertForSequenceClassification model
author: John Snow Labs
name: bert_classifier_finbert_esg
date: 2022-06-30
tags: [open_source, bert, classification, finance, esg, en]
task: Text Classification
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Text Classification model for detecting Environment, Social and Governance-related texts, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `finbert-esg` is a English model originally trained by `yiyanghkust`.

Classes: `None`, `Environmental`, `Social`, `Governance`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_finbert_esg_en_4.0.0_3.0_1656600149268.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
sequenceClassifier = BertForSequenceClassification.pretrained("bert_classifier_finbert_esg","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["I love Spark NLP."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_classifier_finbert_esg","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("I love Spark NLP.").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_finbert_esg|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

https://huggingface.co/yiyanghkust/finbert-esg
