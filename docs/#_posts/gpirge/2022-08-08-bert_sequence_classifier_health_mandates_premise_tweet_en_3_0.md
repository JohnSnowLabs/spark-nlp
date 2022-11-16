---
layout: model
title: Premise About Health Mandates Related to Covid-19 Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_health_mandates_premise_tweet
date: 2022-08-08
tags: [en, clinical, licensed, public_health, classifier, sequence_classification, covid_19, tweet, premise, mandate]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify premise about health mandates related to Covid-19 from tweets. 
This model is intended for direct use as a classification model and the target classes are: Has no premise (no argument), Has premise (argument).

## Predicted Entities

`Has premise (argument)`, `Has no premise (no argument)`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_MANDATES/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mandates_premise_tweet_en_4.0.2_3.0_1659983362057.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mandates_premise_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["""It's too dangerous to hold the RNC, but let's send students and teachers back to school.""",
"""So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments You're Speaker Pelosi nephew so stop the agenda LIES.""",
"""Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other.""",
"""Go to a restaurant no mask Do a food shop wear a mask INCONSISTENT No Masks No Masks.""",
"""But if schools close who is gonna occupy those graves Cause politiciansprotected smokers protected drunkardsprotected school kids amp teachers""",
"""New title Maskhole I think I'm going to use this very soon coronavirus."""
], StringType()).toDF("text")
                              
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mandates_premise_tweet", "es", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("It's too dangerous to hold the RNC, but let's send students and teachers back to school.",
                     "So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments You're Speaker Pelosi nephew so stop the agenda LIES.",
                     "Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other.",
                     "Go to a restaurant no mask Do a food shop wear a mask INCONSISTENT No Masks No Masks.",
                     "But if schools close who is gonna occupy those graves Cause politiciansprotected smokers protected drunkardsprotected school kids amp teachers",
                     "New title Maskhole I think I'm going to use this very soon coronavirus.")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------+
|text                                                                                                                                                              |result                        |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------+
|It's too dangerous to hold the RNC, but let's send students and teachers back to school.                                                                          |[Has premise (argument)]      |
|So is the flu and pneumonia what are their s stop the Media Manipulation covid has treatments You're Speaker Pelosi nephew so stop the agenda LIES.               |[Has premise (argument)]      |
|Just a quick update to my U.S. followers, I'll be making a stop in all 50 states this spring!  No tickets needed, just don't wash your hands, cough on each other.|[Has no premise (no argument)]|
|Go to a restaurant no mask Do a food shop wear a mask INCONSISTENT No Masks No Masks.                                                                             |[Has no premise (no argument)]|
|But if schools close who is gonna occupy those graves Cause politiciansprotected smokers protected drunkardsprotected school kids amp teachers                    |[Has premise (argument)]      |
|New title Maskhole I think I'm going to use this very soon coronavirus.                                                                                           |[Has no premise (no argument)]|
+------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mandates_premise_tweet|
|Compatibility:|Healthcare NLP 4.0.2+|
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
Has_no_premise_(no_argument)     0.9000    0.6992    0.7870       502
      Has_premise_(argument)     0.6576    0.8815    0.7532       329
                    accuracy     -         -         0.7714       831
                   macro-avg     0.7788    0.7903    0.7701       831
                weighted-avg     0.8040    0.7714    0.7736       831
```