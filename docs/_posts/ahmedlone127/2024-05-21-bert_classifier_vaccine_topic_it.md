---
layout: model
title: Italian BertForSequenceClassification Cased model (from brema76)
author: John Snow Labs
name: bert_classifier_vaccine_topic
date: 2024-05-21
tags: [bert, sequence_classification, classification, open_source, it, onnx]
task: Text Classification
language: it
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `vaccine_topic_it` is a Italian model originally trained by `brema76`.

## Predicted Entities

`Effectiveness of vaccination`, `Vaccine business`, `Legal issues`, `Other`, `Administration of vaccines`, `Safety concerns`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_vaccine_topic_it_5.2.4_3.0_1716291860176.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_vaccine_topic_it_5.2.4_3.0_1716291860176.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_vaccine_topic","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_vaccine_topic","it") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Adoro Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("it.classify.bert.vaccine.").predict("""Adoro Spark NLP""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_vaccine_topic|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|it|
|Size:|667.3 MB|

## References

References

- https://huggingface.co/brema76/vaccine_topic_it
- https://arxiv.org/abs/2207.12264