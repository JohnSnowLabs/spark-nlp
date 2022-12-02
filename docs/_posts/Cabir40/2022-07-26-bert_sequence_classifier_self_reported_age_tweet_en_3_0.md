---
layout: model
title: Self Report Age Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_self_reported_age_tweet
date: 2022-07-26
tags: [licensed, clinical, en, classifier, sequence_classification, age, public_health]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify self-report the exact age into social media data.

## Predicted Entities

`self_report_age`, `no_report`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_AGE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_age_tweet_en_4.0.0_3.0_1658852070357.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_age_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["Who knew I would spend my Saturday mornings at 21 still watching Disney channel",
                              "My girl, Fancy, just turned 17. She’s getting up there, but she still has the energy of a puppy"], StringType()).toDF("text")
                              
result = pipeline.fit(data).transform(data)

# Checking results
result.select("text", "class.result").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_age_tweet", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("Who knew I would spend my Saturday mornings at 21 still watching Disney channel",
                      "My girl, Fancy, just turned 17. She’s getting up there, but she still has the energy of a puppy")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------+-----------------+
|text                                                                                           |result           |
+-----------------------------------------------------------------------------------------------+-----------------+
|Who knew I would spend my Saturday mornings at 21 still watching Disney channel                |[self_report_age]|
|My girl, Fancy, just turned 17. She’s getting up there, but she still has the energy of a puppy|[no_report]      |
+-----------------------------------------------------------------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_age_tweet|
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
          label  precision    recall  f1-score  support
      no_report   0.939016  0.900332  0.919267     1505
self_report_age   0.801849  0.873381  0.836088      695
       accuracy   -         -         0.891818     2200
      macro-avg   0.870433  0.886857  0.877678     2200
   weighted-avg   0.895684  0.891818  0.892990     2200
```
