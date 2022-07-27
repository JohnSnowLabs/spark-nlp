---
layout: model
title: Self Reported Stress Classifier (RoBerta)
author: John Snow Labs
name: robert_sequence_classifier_self_reported_stress_tweet
date: 2022-07-27
tags: [en, public_health, licenced, classifier, stress, roberta, clinical, sequence_classification, licensed]
task: Text Classification
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a RoBerta sequence classifier that can identify stress in social media (Twitter) posts in the self-disclosure category. `0` class is `not-stressed` while `1` is `stressed`.

## Predicted Entities

`0`, `1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/robert_sequence_classifier_self_reported_stress_tweet_en_4.0.0_3.0_1658920611223.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = RoBertaForSequenceClassification.pretrained("robert_sequence_classifier_self_reported_stress_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([["Do you feel stressed!"], ["I'm so stressed!"],
                              ["Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful."], 
                              ["@User Do you enjoy living constantly in this self-inflicted stress?"]]).toDF("text")
                             
result = pipeline.fit(data).transform(data)

result.select("class.result", "text").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("token")

val sequenceClassifier = RoBertaForSequenceClassification.pretrained("robert_sequence_classifier_self_reported_stress_tweet", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("Do you feel stressed!", "I'm so stressed!",
                      "Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful.", 
                      "@User Do you enjoy living constantly in this self-inflicted stress?")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------+-------------------------------------------------------------------------------------------------------------+
|result|text                                                                                                         |
+------+-------------------------------------------------------------------------------------------------------------+
|[0]   |Do you feel stressed!                                                                                        |
|[1]   |I'm so stressed!                                                                                             |
|[1]   |Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful.|
|[0]   |@User Do you enjoy living constantly in this self-inflicted stress?                                          |
+------+-------------------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robert_sequence_classifier_self_reported_stress_tweet|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
       label  precision    recall  f1-score   support
           0       0.93      0.88      0.90       427
           1       0.81      0.88      0.84       245
    accuracy         -         -       0.88       672
   macro-avg       0.87      0.88      0.87       672
weighted-avg       0.88      0.88      0.88       672
```