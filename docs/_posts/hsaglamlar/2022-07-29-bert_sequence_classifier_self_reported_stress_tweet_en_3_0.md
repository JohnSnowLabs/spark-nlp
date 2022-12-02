---
layout: model
title: Self Reported Stress Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_self_reported_stress_tweet
date: 2022-07-29
tags: [en, licenced, clinical, public_health, sequence_classification, classifier, stress, licensed]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can identify stress in social media (Twitter) posts in the self-disclosure category. The model finds whether a person claims he/she is stressed or not.

## Predicted Entities

`not-stressed`, `stressed`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_STRESS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_stress_tweet_en_4.0.0_3.0_1659087442993.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_stress_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([["Do you feel stressed?"], 
                              ["I'm so stressed!"],
                              ["Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful."], 
                              ["Do you enjoy living constantly in this self-inflicted stress?"]]).toDF("text")
                              
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_stress_tweet", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val data = Seq(Array("Do you feel stressed!", 
                     "I'm so stressed!",
                     "Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful.", 
                     "Do you enjoy living constantly in this self-inflicted stress?")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------------------+--------------+
|text                                                                                                         |result        |
+-------------------------------------------------------------------------------------------------------------+--------------+
|Do you feel stressed?                                                                                        |[not-stressed]|
|I'm so stressed!                                                                                             |[stressed]    |
|Depression and anxiety will probably end up killing me – I feel so stressed all the time and just feel awful.|[stressed]    |
|Do you enjoy living constantly in this self-inflicted stress?                                                |[not-stressed]|
+-------------------------------------------------------------------------------------------------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_stress_tweet|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
       label  precision    recall  f1-score   support
not-stressed     0.8564    0.8020    0.8283       409
    stressed     0.7197    0.7909    0.7536       263
    accuracy        -         -      0.7976       672
   macro-avg     0.7881    0.7964    0.7910       672
weighted-avg     0.8029    0.7976    0.7991       672

```
