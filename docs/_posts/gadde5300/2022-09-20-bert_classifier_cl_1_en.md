---
layout: model
title: English BertForSequenceClassification Cased model (from yuan1729)
author: John Snow Labs
name: bert_classifier_cl_1
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `CL_1` is a English model originally trained by `yuan1729`.

## Predicted Entities

`搶奪強盜及海盜罪`, `藏匿人犯及湮滅證據罪`, `賭博罪`, `侵占罪`, `遺棄罪`, `恐嚇及擄人勒贖罪`, `殺人罪`, `妨害秩序罪`, `偽證及誣告罪`, `妨害電腦使用罪`, `妨害風化罪`, `瀆職罪`, `妨害婚姻及家庭罪`, `竊盜罪`, `妨害名譽及信用罪`, `傷害罪`, `妨害性自主罪`, `贓物罪`, `妨害自由罪`, `妨害秘密罪`, `妨害公務罪`, `詐欺背信及重利罪`, `妨害投票罪`, `偽造文書印文罪`, `偽造有價證券罪`, `公共危險罪`, `毀棄損壞罪`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_cl_1_en_4.2.0_3.0_1663666869094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_cl_1_en_4.2.0_3.0_1663666869094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_cl_1","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_cl_1","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_cl_1|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|383.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/yuan1729/CL_1