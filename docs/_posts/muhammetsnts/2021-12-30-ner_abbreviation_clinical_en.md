---
layout: model
title: Extraction of Clinical Abbreviations and Acronyms
author: John Snow Labs
name: ner_abbreviation_clinical
date: 2021-12-30
tags: [ner, abbreviation, acronym, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to extract clinical abbreviations and acronyms in text.

## Predicted Entities

`ABBR`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ABBREVIATION/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_abbreviation_clinical_en_3.3.4_2.4_1640852436967.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_abbreviation_clinical_en_3.3.4_2.4_1640852436967.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")\

embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
.setInputCols(['sentence', 'token']) \
.setOutputCol('embeddings')

abbr_ner = MedicalNerModel.pretrained('ner_abbreviation_clinical', 'en', 'clinical/models') \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("abbr_ner")

abbr_converter = NerConverter() \
.setInputCols(["sentence", "token", "abbr_ner"]) \
.setOutputCol("abbr_ner_chunk")\


ner_pipeline = Pipeline(
stages = [
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
abbr_ner,
abbr_converter
])

sample_df = spark.createDataFrame([["Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."]]).toDF("text")
result = ner_pipeline.fit(sample_df).transform(sample_df)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("embeddings")

val abbr_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") 
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("abbr_ner")

val abbr_converter = NerConverter() 
.setInputCols(Array("sentence", "token", "abbr_ner")) 
.setOutputCol("abbr_ner_chunk")


val ner_pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, abbr_ner, abbr_converter))

val sample_df = Seq("Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.").toDF("text")
val result = ner_pipeline.fit(sample_df).transform(sample_df)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.abbreviation_clinical").predict("""Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.""")
```

</div>

## Results

```bash
+-----+---------+
|chunk|ner_label|
+-----+---------+
|CBC  |ABBR     |
|AB   |ABBR     |
|VDRL |ABBR     |
|HIV  |ABBR     |
+-----+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_abbreviation_clinical|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.6 MB|

## Data Source

Trained on the in-house dataset.

## Benchmarking

```bash
Quality on validation dataset (20.0%), validation examples = 454
time to finish evaluation: 5.34s

+-------+------+------+------+----------+------+------+
| Label |    tp|    fp|    fn| precision|recall|    f1|
+-------+------+------+------+----------+------+------+
| B-ABBR| 672.0|  42.0|  40.0|    0.9411|0.9438|0.9424|
+-------+------+------+------+----------+------+------+

+------------+----------+--------+--------+
|            | precision|  recall|      f1|
+------------+----------+--------+--------+
|       macro|    0.9411|  0.9438|  0.9424|
+------------+----------+--------+--------+
|       micro|    0.9411|  0.9438|  0.9424|
+------------+----------+--------+--------+
```
