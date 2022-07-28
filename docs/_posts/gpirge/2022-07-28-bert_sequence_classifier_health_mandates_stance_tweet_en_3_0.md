---
layout: model
title: Stance About Health Mandates Related to Covid-19 Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_health_mandates_stance_tweet
date: 2022-07-28
tags: [en, clinical, licensed, public_health, classifier, sequence_classification, covid_19, tweet, stance, mandate]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify stance about health mandates related to Covid-19 from tweets. 
This model is intended for direct use as a classification model and the target classes are: FAVOR, AGAINST, NONE.

## Predicted Entities

`FAVOR`, `AGAINST`, `NONE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mandates_stance_tweet_en_4.0.0_3.0_1659012324029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mandates_stance_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["It's too dangerous to hold the RNC, but let's send students and teachers back to school",
                              "So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments Youre Speaker Pelosi nephew so stop the agenda LIES",
                              "Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other"], StringType()).toDF("text")
                              
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mandates_stance_tweet", "es", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("It's too dangerous to hold the RNC, but let's send students and teachers back to school",
                     "So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments Youre Speaker Pelosi nephew so stop the agenda LIES",
                     "Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+
|text                                                                                                                                                             |result   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+
|It's too dangerous to hold the RNC, but let's send students and teachers back to school                                                                          |[FAVOR]  |
|So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments Youre Speaker Pelosi nephew so stop the agenda LIES                |[AGAINST]|
|Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other|[NONE]   |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mandates_stance_tweet|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

The dataset is Covid-19-specific and consists of tweets collected via a series of keywords associated with that disease.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
     AGAINST       0.63      0.71      0.66       158
       FAVOR       0.76      0.77      0.76       244
        NONE       0.74      0.64      0.69       197
    accuracy       -         -         0.71       599
   macro-avg       0.71      0.71      0.71       599
weighted-avg       0.72      0.71      0.71       599
```