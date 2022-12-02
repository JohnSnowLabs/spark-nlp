---
layout: model
title: Depression Classifier (PHS-BERT) for Tweets
author: John Snow Labs
name: bert_sequence_classifier_depression_twitter
date: 2022-08-09
tags: [public_health, en, licensed, sequence_classification, mental_health, depression, twitter]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based tweet classification model that can classify whether tweets contain depressive text.

## Predicted Entities

`depression`, `no-depression`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_depression_twitter_en_4.0.2_3.0_1660051816827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
# Sample Python Code

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression_twitter", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
      ["Do what makes you happy, be with who makes you smile, laugh as much as you breathe, and love as long as you live!"],
      ["Everything is a lie, everyone is fake, I'm so tired of living"]
    ]).toDF("text")


result = pipeline.fit(data).transform(data)
result.select("text", "class.result").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression_twitter", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array(
         "Do what makes you happy, be with who makes you smile, laugh as much as you breathe, and love as long as you live!",
         "Everything is a lie, everyone is fake, I'm so tired of living"
    )).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------+---------------+
|text                                                                                                             |result         |
+-----------------------------------------------------------------------------------------------------------------+---------------+
|Do what makes you happy, be with who makes you smile, laugh as much as you breathe, and love as long as you live!|[no-depression]|
|Everything is a lie, everyone is fake, I'm so tired of living                                                    |[depression]   |
+-----------------------------------------------------------------------------------------------------------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_depression_twitter|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
          label   precision    recall  f1-score   support 
        minimum       0.97      0.98      0.97      1411 
high-depression       0.95      0.92      0.93       595 
       accuracy        -          -       0.96      2006 
      macro-avg       0.96      0.95      0.95      2006 
   weighted-avg       0.96      0.96      0.96      2006
```