---
layout: model
title: Detect Adverse Drug Events (MedicalBertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_ade_binary
date: 2022-07-27
tags: [clinical, ade, licensed, public_health, token_classification, ner, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect adverse reactions of drugs in texts excahnged over twitter. This model is trained with the `BertForTokenClassification` method from the transformers library and imported into Spark NLP.

## Predicted Entities

`ADE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_ade_binary_en_4.0.0_3.0_1658906410540.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_ade_binary_en_4.0.0_3.0_1658906410540.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_ade_binary", "en", "clinical/models")\
    .setInputCols("token", "sentence")\
    .setOutputCol("label")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","label"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["I used to be on paxil but that made me more depressed and prozac made me angry",
                              "Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs."], StringType()).toDF("text")

result = model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_ade_binary", "en", "clinical/models")
  .setInputCols(Array("token", "sentence"))
  .setOutputCol("label")
  .setCaseSensitive(True)

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence", "token", "label"))
  .setOutputCol("ner_chunk")


val pipeline =  new Pipeline().setStages(Array(
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter))

val data = Seq(Array("I used to be on paxil but that made me more depressed and prozac made me angry",
                     "Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs.")).toDS().toDF("text")

val result = model.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+---------+
|chunk        |ner_label|
+-------------+---------+
|depressed    |ADE      |
|angry        |ADE      |
|sugar crashes|ADE      |
+-------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_ade_binary|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
       label  precision    recall  f1-score   support
       B-ADE       0.89      0.88      0.88      3720
       I-ADE       0.85      0.84      0.84      3145
           O       0.98      0.98      0.98     26963
    accuracy        -         -        0.95     33828
   macro-avg       0.90      0.90      0.90     33828
weighted-avg       0.95      0.95      0.95     33828
```
