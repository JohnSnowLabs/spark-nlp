---
layout: model
title: Classification of Vaccine Tweets
author: John Snow Labs
name: bert_sequence_classifier_self_reported_vaccine_status_tweet
date: 2022-07-27
tags: [bert_sequence_classifier, bert, en, licensed, vaccine, clinical, public_health, tweet, classifier, sequence_classification]
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

Classification of tweets indicating self-reported COVID-19 vaccination status. This model involves the identification of self-reported COVID-19 vaccination status in English tweets. ([SMM4H 2022](https://healthlanguageprocessing.org/smm4h-2022/))

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_vaccine_status_tweet_en_3.5.0_3.0_1658928804893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.load(f'downloaded_models/{MODEL_NEW_NAME}')\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

# Generating example
data = spark.createDataFrame(["I came to a point finally and i've vaccinated, didnt feel pain.Suggest everyone",
                              "If Pfizer believes we need a booster shot, we need it. Who knows their product better? Following the guidance of @CDCgov is how I wound up w/ Covid-19 and having to shut down my K-2 classroom for an entire week. I will do whatever it takes to protect my students, friends, family."], StringType()).toDF("text")
                              
result = pipeline.fit(data).transform(data)

# Checking results
result.select("text", "class.result").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_vaccine_status_tweet", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline.setStages(Array(document_assembler, tokenizer, sequenceClassifier))

# couple of simple examples
val example = Seq("""I came to a point finally and i've vaccinated, didnt feel pain.Suggest everyone",
                    "If Pfizer believes we need a booster shot, we need it. Who knows their product better? Following the guidance of @CDCgov is how I wound up w/ Covid-19 and having to shut down my K-2 classroom for an entire week. I will do whatever it takes to protect my students, friends, family.""").toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
|text                                                                                                                                                                                                                                                                                    |result           |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
|I came to a point finally and i've vaccinated, didnt feel pain.Suggest everyone                                                                                                                                                                                                         |[Self_reports]   |
|If Pfizer believes we need a booster shot, we need it. Who knows their product better? Following the guidance of @CDCgov is how I wound up w/ Covid-19 and having to shut down my K-2 classroom for an entire week. I will do whatever it takes to protect my students, friends, family.|[Vaccine_chatter]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_vaccine_status_tweet|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
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
   Self_reports       0.79      0.78      0.78       311
Vaccine_chatter       0.97      0.97      0.97      2410
       accuracy        -         -        0.95      2721
      macro-avg       0.88      0.88      0.88      2721
   weighted-avg       0.95      0.95      0.95      2721
```
