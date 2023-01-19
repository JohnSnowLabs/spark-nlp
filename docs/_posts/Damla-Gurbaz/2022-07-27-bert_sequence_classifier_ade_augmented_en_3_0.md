---
layout: model
title: Adverse Drug Events Binary Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_ade_augmented
date: 2022-07-27
tags: [clinical, licensed, public_health, ade, classifier, sequence_classification, en]
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

This model is a [BioBERT-based] (https://github.com/dmis-lab/biobert) classifier that can classify tweets reporting ADEs (Adverse Drug Events).

## Predicted Entities

`ADE`, `noADE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_ade_augmented_en_4.0.0_3.0_1658905698079.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_ade_augmented_en_4.0.0_3.0_1658905698079.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_ade_augmented", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["So glad I am off effexor, so sad it ruined my teeth. tip Please be carefull taking antideppresiva and read about it 1st",
                              "Religare Capital Ranbaxy has been accepting approval for Diovan since 2012"], StringType()).toDF("text")
              
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_ade_augmented", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("So glad I am off effexor, so sad it ruined my teeth. tip Please be carefull taking antideppresiva and read about it 1st",
                     "Religare Capital Ranbaxy has been accepting approval for Diovan since 2012")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------------+-------+
|text                                                                                                                   |result |
+-----------------------------------------------------------------------------------------------------------------------+-------+
|So glad I am off effexor, so sad it ruined my teeth. tip Please be carefull taking antideppresiva and read about it 1st|[ADE]  |
|Religare Capital Ranbaxy has been accepting approval for Diovan since 2012                                             |[noADE]|
+-----------------------------------------------------------------------------------------------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_ade_augmented|
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
         ADE     0.9696    0.9595    0.9645      2763
       noADE     0.9670    0.9753    0.9712      3366
    accuracy       -         -       0.9682      6129
   macro-avg     0.9683    0.9674    0.9678      6129
weighted-avg     0.9682    0.9682    0.9682      6129
```
