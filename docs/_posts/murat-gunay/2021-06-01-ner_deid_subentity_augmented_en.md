---
layout: model
title: Detect PHI for Deidentification (Subentity- Augmented)
author: John Snow Labs
name: ner_deid_subentity_augmented
date: 2021-06-01
tags: [en, clinical, ner, licensed, deid]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. Deidentification NER (Absolute) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. It detects 23 entities. This ner model is trained with combination of i2b2 train set and augmented version of i2b2 train set.

## Predicted Entities

`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_en_3.0.3_3.0_1622539925891.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk_subentity")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""]})))
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(Array("sentence", "token"))\
      .setOutputCol("embeddings")

val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(Array("sentence", "token", "embeddings")) \
      .setOutputCol("ner")

val ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk_subentity")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

val result = pipeline.fit(Seq.empty["A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."].toDS.toDF("text")).transform(data)
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
|25-year-old                  |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street.           |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_augmented|
|Compatibility:|Spark NLP for Healthcare 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

A custom data set which is created from the i2b2-PHI train and the augmented version of the i2b2-PHI train set is used.

## Benchmarking

```bash
+-------------+------+-----+-----+------+---------+------+------+
|       entity|    tp|   fp|   fn| total|precision|recall|    f1|
+-------------+------+-----+-----+------+---------+------+------+
|      PATIENT|1468.0|196.0|154.0|1622.0|   0.8822|0.9051|0.8935|
|     HOSPITAL|1400.0|135.0|182.0|1582.0|   0.9121| 0.885|0.8983|
|         DATE|5546.0| 68.0| 96.0|5642.0|   0.9879| 0.983|0.9854|
| ORGANIZATION|  65.0| 22.0| 73.0| 138.0|   0.7471| 0.471|0.5778|
|         CITY| 279.0| 71.0| 62.0| 341.0|   0.7971|0.8182|0.8075|
|       STREET| 407.0| 11.0|  8.0| 415.0|   0.9737|0.9807|0.9772|
|     USERNAME|  89.0|  4.0| 12.0| 101.0|    0.957|0.8812|0.9175|
|       DEVICE|   0.0|  0.0| 24.0|  24.0|      0.0|   0.0|   0.0|
|          FAX|   0.0|  1.0|  6.0|   6.0|      0.0|   0.0|   0.0|
|        IDNUM| 159.0| 31.0| 57.0| 216.0|   0.8368|0.7361|0.7833|
|        STATE| 160.0| 28.0| 45.0| 205.0|   0.8511|0.7805|0.8142|
|        EMAIL|   1.0|  0.0|  0.0|   1.0|      1.0|   1.0|   1.0|
|          ZIP| 134.0|  9.0|  5.0| 139.0|   0.9371| 0.964|0.9504|
|MEDICALRECORD| 405.0| 19.0| 38.0| 443.0|   0.9552|0.9142|0.9343|
|        OTHER|   0.0|  2.0| 22.0|  22.0|      0.0|   0.0|   0.0|
|   PROFESSION| 234.0| 23.0|139.0| 373.0|   0.9105|0.6273|0.7429|
|        PHONE| 334.0| 21.0| 14.0| 348.0|   0.9408|0.9598|0.9502|
|      COUNTRY|  77.0| 13.0| 51.0| 128.0|   0.8556|0.6016|0.7064|
|   HEALTHPLAN|   0.0|  0.0|  2.0|   2.0|      0.0|   0.0|   0.0|
|       DOCTOR|3280.0|191.0|267.0|3547.0|    0.945|0.9247|0.9347|
|          AGE| 714.0| 32.0| 48.0| 762.0|   0.9571| 0.937|0.9469|
+-------------+------+-----+-----+------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.7057395025229919|
+------------------+

+------------------+
|             micro|
+------------------+
|0.9280564484355146|
+------------------+
```