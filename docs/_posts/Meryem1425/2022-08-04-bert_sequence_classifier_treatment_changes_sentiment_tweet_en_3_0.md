---
layout: model
title: Self Treatment Changes Classifier in Tweets (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_treatment_changes_sentiment_tweet
date: 2022-08-04
tags: [en, clinical, licensed, public_health, classifier, sequence_classification, treatment_changes, sentiment, treatment]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify patients non-adherent to their treatments and their reasons on Twitter.

## Predicted Entities

`negative`, `positive`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_CHANGE_DRUG_TREATMENT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_treatment_changes_sentiment_tweet_en_4.0.2_3.0_1659637869883.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_treatment_changes_sentiment_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["I love when they say things like this. I took that ambien instead of my thyroid pill.",
                              "I am a 30 year old man who is not overweight but is still on the verge of needing a Lipitor prescription."], StringType()).toDF("text")
                          
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_treatment_changes_sentiment_tweet", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val data = Seq(Array("I love when they say things like this. I took that ambien instead of my thyroid pill.",
                      "I am a 30 year old man who is not overweight but is still on the verge of needing a Lipitor prescription.")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------+----------+
|text                                                                                                     |result    |
+---------------------------------------------------------------------------------------------------------+----------+
|I love when they say things like this. I took that ambien instead of my thyroid pill.                    |[positive]|
|I am a 30 year old man who is not overweight but is still on the verge of needing a Lipitor prescription.|[negative]|
+---------------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_treatment_changes_sentiment_tweet|
|Compatibility:|Healthcare NLP 4.0.2+|
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
    negative     0.9515    0.9751    0.9632      1368
    positive     0.6304    0.4603    0.5321       126
    accuracy     -         -         0.9317      1494
   macro-avg     0.7910    0.7177    0.7476      1494
weighted-avg     0.9244    0.9317    0.9268      1494
```