---
layout: model
title: Self-Reported Covid-19 Symptoms Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_self_reported_symptoms_tweet
date: 2022-07-28
tags: [es, clinical, licensed, public_health, classifier, sequence_classification, covid_19, tweet, symptom]
task: Text Classification
language: es
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [BERT based](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) classifier that can classify the origin of symptoms related to Covid-19 from Spanish tweets. 
This model is intended for direct use as a classification model and the target classes are: Lit-News_mentions, Self_reports, non-personal_reports.

## Predicted Entities

`Lit-News_mentions`, `Self_reports`, `non-personal_reports`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_COVID_SYMPTOMS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4TC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_symptoms_tweet_es_4.0.0_3.0_1659022252550.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_symptoms_tweet_es_4.0.0_3.0_1659022252550.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_symptoms_tweet", "es", "clinical/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["Las vacunas 3 y hablamos inminidad vivo  Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃorgicas si que sepan",
                              "Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.",
                              "Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus"], StringType()).toDF("text")
result = model.transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_symptoms_tweet", "es", "clinical/models")
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512)

val pipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                   tokenizer,
                                                   sequenceClassifier
                                                   ))

val data = Seq(Array("Las vacunas 3 y hablamos inminidad vivo  Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃorgicas si que sepan",
                     "Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.",
                     "Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus")).toDS.toDF("text")
val result = model.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
|text                                                                                                                                             |result                |
+-------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
|Las vacunas 3 y hablamos inminidad vivo  Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃorgicas si que sepan         |[non-personal_reports]|
|Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.|[Self_reports]        |
|Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus                                                    |[Lit-News_mentions]   |
+-------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_symptoms_tweet|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
               label  precision    recall  f1-score   support
   Lit-News_mentions       0.93      0.95      0.94       309
        Self_reports       0.65      0.74      0.69        72
non-personal_reports       0.79      0.67      0.73       122
            accuracy       -         -         0.85       503
           macro-avg       0.79      0.79      0.79       503
        weighted-avg       0.85      0.85      0.85       503
```
