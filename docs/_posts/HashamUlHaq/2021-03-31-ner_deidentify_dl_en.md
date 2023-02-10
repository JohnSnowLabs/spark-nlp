---
layout: model
title: Detect PHI for Deidentification (ner_deidentification_dl)
author: John Snow Labs
name: ner_deidentify_dl
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition annotator (NERDLModel) allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.Deidentification NER (DL) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified.

## Predicted Entities

`Age`,`BIOID`,`City`,`Country`,`Country`,`Date`,`Device`,`Doctor`,`EMail`,`Hospital`,`Fax`,`Healthplan`,`Hospital`,,`Idnum`,`Location-Other`,`Medicalrecord`,`Organization`,`Patient`,`Phone`,`Profession`,`State`,`Street`,`URL`,`Username`,`Zip`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deidentify_dl_en_3.0.0_3.0_1617209710705.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deidentify_dl_en_3.0.0_3.0_1617209710705.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_deidentify_dl","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 month years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street"]], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_deidentify_dl","en","clinical/models")
	.setInputCols(Array("sentence","token","embeddings"))
	.setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 month years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.deid").predict("""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 month years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street""")
```

</div>

## Results

```bash
+---------------+-----+
|ner_label      |count|
+---------------+-----+
|O              |28   |
|I-HOSPITAL     |4    |
|B-DATE         |3    |
|I-STREET       |3    |
|I-PATIENT      |2    |
|B-DOCTOR       |2    |
|B-AGE          |1    |
|B-PATIENT      |1    |
|I-DOCTOR       |1    |
|B-MEDICALRECORD|1    |
+---------------+-----+. 

+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson , Ora            |PATIENT      |
|7194334                      |MEDICALRECORD|
|01/13/93                     |DATE         |
|Oliveira                     |DOCTOR       |
|25                           |AGE          |
|2079-11-09                   |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
+-----------------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deidentify_dl|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on JSL enriched n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with embeddings_clinical https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

## Benchmarking

```bash
|    | label            |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|:-----------------|------:|-----:|-----:|---------:|---------:|---------:|
|  1 | I-AGE            |     7 |    3 |    6 | 0.7      | 0.538462 | 0.608696 |
|  2 | I-DOCTOR         |   800 |   27 |   94 | 0.967352 | 0.894855 | 0.929692 |
|  3 | I-IDNUM          |     6 |    0 |    2 | 1        | 0.75     | 0.857143 |
|  4 | B-DATE           |  1883 |   34 |   56 | 0.982264 | 0.971119 | 0.97666  |
|  5 | I-DATE           |   425 |   28 |   25 | 0.93819  | 0.944444 | 0.941307 |
|  6 | B-PHONE          |    29 |    7 |    9 | 0.805556 | 0.763158 | 0.783784 |
|  7 | B-STATE          |    87 |    4 |   11 | 0.956044 | 0.887755 | 0.920635 |
|  8 | B-CITY           |    35 |   11 |   26 | 0.76087  | 0.57377  | 0.654206 |
|  9 | I-ORGANIZATION   |    12 |    4 |   15 | 0.75     | 0.444444 | 0.55814  |
| 10 | B-DOCTOR         |   728 |   75 |   53 | 0.9066   | 0.932138 | 0.919192 |
| 11 | I-PROFESSION     |    43 |   11 |   13 | 0.796296 | 0.767857 | 0.781818 |
| 12 | I-PHONE          |    62 |    4 |    4 | 0.939394 | 0.939394 | 0.939394 |
| 13 | B-AGE            |   234 |   13 |   16 | 0.947368 | 0.936    | 0.94165  |
| 14 | B-STREET         |    20 |    7 |   16 | 0.740741 | 0.555556 | 0.634921 |
| 15 | I-ZIP            |    60 |    3 |    2 | 0.952381 | 0.967742 | 0.96     |
| 16 | I-MEDICALRECORD  |    54 |    5 |    2 | 0.915254 | 0.964286 | 0.93913  |
| 17 | B-ZIP            |     2 |    1 |    0 | 0.666667 | 1        | 0.8      |
| 18 | B-HOSPITAL       |   256 |   23 |   66 | 0.917563 | 0.795031 | 0.851913 |
| 19 | I-STREET         |   150 |   17 |   20 | 0.898204 | 0.882353 | 0.890208 |
| 20 | B-COUNTRY        |    22 |    2 |    8 | 0.916667 | 0.733333 | 0.814815 |
| 21 | I-COUNTRY        |     1 |    0 |    0 | 1        | 1        | 1        |
| 22 | I-STATE          |     6 |    0 |    1 | 1        | 0.857143 | 0.923077 |
| 23 | B-USERNAME       |    30 |    0 |    4 | 1        | 0.882353 | 0.9375   |
| 24 | I-HOSPITAL       |   295 |   37 |   64 | 0.888554 | 0.821727 | 0.853835 |
| 25 | I-PATIENT        |   243 |   26 |   41 | 0.903346 | 0.855634 | 0.878843 |
| 26 | B-PROFESSION     |    52 |    8 |   17 | 0.866667 | 0.753623 | 0.806202 |
| 27 | B-IDNUM          |    32 |    3 |   12 | 0.914286 | 0.727273 | 0.810127 |
| 28 | I-CITY           |    76 |   15 |   13 | 0.835165 | 0.853933 | 0.844444 |
| 29 | B-PATIENT        |   337 |   29 |   40 | 0.920765 | 0.893899 | 0.907133 |
| 30 | B-MEDICALRECORD  |    74 |    6 |    4 | 0.925    | 0.948718 | 0.936709 |
| 31 | B-ORGANIZATION   |    20 |    5 |   13 | 0.8      | 0.606061 | 0.689655 |
| 32 | Macro-average    | 6083  | 408  |  673 | 0.7976   | 0.697533 | 0.744218 |
| 33 | Micro-average    | 6083  | 408  |  673 | 0.937144 | 0.900385 | 0.918397 |
```