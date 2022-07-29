---
layout: model
title: Premise About Health Mandates Related to Covid-19 Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_health_mandates_premise_tweet
date: 2022-07-29
tags: [en, clinical, licensed, public_health, classifier, sequence_classification, covid_19, tweet, premise, mandate]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify premise about health mandates related to Covid-19 from tweets. 
This model is intended for direct use as a classification model and the target classes are: has_no_premise, has_premise.

## Predicted Entities

`has_premise`, `has_no_premise`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mandates_premise_tweet_en_4.0.0_3.0_1659112971420.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

data = spark.createDataFrame(["Schools need to reopen in some form here in the fall. If for no other reason than for many of our students we are the most responsible and dependable adults they will see all day.",
                              "Had a crazy dream that the 'rona virus was prepping our bodies for a zombie virus. What a nightmare!"], StringType()).toDF("text")
                              
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

val data = Seq(Array("Schools need to reopen in some form here in the fall. If for no other reason than for many of our students we are the most responsible and dependable adults they will see all day.",
                              "Had a crazy dream that the 'rona virus was prepping our bodies for a zombie virus. What a nightmare!")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash


+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------+
|text                                                                                                                                                                               |result          |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------+
|Schools need to reopen in some form here in the fall. If for no other reason than for many of our students we are the most responsible and dependable adults they will see all day.|[has_premise]   |
|Had a crazy dream that the 'rona virus was prepping our bodies for a zombie virus. What a nightmare!                                                                               |[has_no_premise]|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mandates_premise_tweet|
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
         label     precision    recall  f1-score   support
has_no_premise     0.8756    0.7215    0.7911       517
   has_premise     0.6444    0.8312    0.7260       314
      accuracy                         0.7629       831
     macro-avg     0.7600    0.7763    0.7586       831
  weighted-avg     0.7882    0.7629    0.7665       831
```