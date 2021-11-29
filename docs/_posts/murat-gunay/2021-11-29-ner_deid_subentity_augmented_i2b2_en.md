---
layout: model
title: Detect PHI for Deidentification
author: John Snow Labs
name: ner_deid_subentity_augmented_i2b2
date: 2021-11-29
tags: [deid, ner, phi, deidentification, licensed, i2b2, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN. 

Deidentification NER is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. It detects 23 entities. This ner model is trained with reviewed version of the re-augmented 2014 i2b2 Deid dataset.

## Predicted Entities

`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_i2b2_en_3.3.2_2.4_1638185564971.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented_i2b2", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk_subentity")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 years old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""]})))
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")

val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented_i2b2", "en", "clinical/models") 
      .setInputCols(Array("sentence", "token", "embeddings")) 
      .setOutputCol("ner")

val ner_converter = NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk_subentity")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))
val model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

val result = pipeline.fit(Seq.empty["A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 years old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|01/13/93                     |DATE         |
|Oliveira                     |DOCTOR       |
|25                           |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine Corp      |ORGANIZATION |
+-----------------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_augmented_i2b2|
|Compatibility:|Spark NLP for Healthcare 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

A custom data set -  an uagmented version of 2014 i2b2 Deid dataset.

## Benchmarking

(on official test set from 2014 i2b2 Deid)

```bash
+-------------+------+----+-----+------+---------+------+------+
|       entity|    tp|  fp|   fn| total|precision|recall|    f1|
+-------------+------+----+-----+------+---------+------+------+
|      PATIENT|1543.0|74.0| 99.0|1642.0|   0.9542|0.9397|0.9469|
|     HOSPITAL|1414.0|66.0|193.0|1607.0|   0.9554|0.8799|0.9161|
|         DATE|5869.0|79.0|134.0|6003.0|   0.9867|0.9777|0.9822|
| ORGANIZATION|  89.0|15.0| 60.0| 149.0|   0.8558|0.5973|0.7036|
|         CITY| 301.0|62.0| 46.0| 347.0|   0.8292|0.8674|0.8479|
|       STREET| 411.0| 1.0|  5.0| 416.0|   0.9976| 0.988|0.9928|
|     USERNAME|  88.0| 0.0|  4.0|  92.0|      1.0|0.9565|0.9778|
|       DEVICE|   9.0| 1.0|  1.0|  10.0|      0.9|   0.9|   0.9|
|        IDNUM| 293.0|25.0| 22.0| 315.0|   0.9214|0.9302|0.9258|
|        STATE| 165.0| 8.0| 42.0| 207.0|   0.9538|0.7971|0.8684|
|          ZIP| 138.0| 3.0|  2.0| 140.0|   0.9787|0.9857|0.9822|
|MEDICALRECORD| 423.0| 8.0| 26.0| 449.0|   0.9814|0.9421|0.9614|
|        OTHER|  10.0| 0.0| 10.0|  20.0|      1.0|   0.5|0.6667|
|   PROFESSION| 284.0|42.0| 56.0| 340.0|   0.8712|0.8353|0.8529|
|        PHONE| 340.0|18.0| 17.0| 357.0|   0.9497|0.9524| 0.951|
|      COUNTRY| 109.0|35.0| 21.0| 130.0|   0.7569|0.8385|0.7956|
|       DOCTOR|3370.0|64.0|248.0|3618.0|   0.9814|0.9315|0.9558|
|          AGE| 742.0|22.0| 29.0| 771.0|   0.9712|0.9624|0.9668|
+-------------+------+----+-----+------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.8096816070245438|
+------------------+

+------------------+
|             micro|
+------------------+
|0.9521127666771618|
+------------------+
```
