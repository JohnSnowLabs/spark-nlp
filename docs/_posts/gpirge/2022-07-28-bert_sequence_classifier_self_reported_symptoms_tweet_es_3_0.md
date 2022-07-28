---
layout: model
title: Self-Reported Covid-19 Symptoms Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_self_reported_symptoms_tweet
date: 2022-07-28
tags: [es, clinical, licensed, public_health, classifier, sequence_classification, covid_19, tweet, symptom, open_source]
task: Text Classification
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
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
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_self_reported_symptoms_tweet_es_4.0.0_3.0_1659012477091.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassifier.pretrained("bert_sequence_classifier_self_reported_symptoms_tweet", "es", "clinical/models")\
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

val sequenceClassifier = MedicalBertForSequenceClassifier.pretrained("bert_sequence_classifier_self_reported_symptoms_tweet", "es", "clinical/models")
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
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

The dataset is Covid-19-specific and consists of Spanish tweets collected via a series of keywords associated with that disease.

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