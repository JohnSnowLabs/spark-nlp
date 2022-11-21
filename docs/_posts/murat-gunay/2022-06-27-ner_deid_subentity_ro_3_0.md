---
layout: model
title: Detect PHI for Deidentification in Romanian (Word2Vec)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-06-27
tags: [ner, deidentification, word2vec, ro, licensed]
task: Named Entity Recognition
language: ro
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.

Deidentification NER (Romanian) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 17 entities.

This NER model is trained with a combination of custom datasets with several data augmentation mechanisms.

## Predicted Entities

`AGE`, `CITY`, `COUNTRY`, `DATE`, `DOCTOR`, `EMAIL`, `FAX`, `HOSPITAL`, `IDNUM`, `LOCATION-OTHER`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `ZIP`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_ro_4.0.0_3.0_1656316441636.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","ro")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "ro", "clinical/models")\
    .setInputCols(["sentence","token","word_embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter])

text = """
Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401"""

data = spark.createDataFrame([[text]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","ro")
.setInputCols(Array("sentence","token"))
.setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "ro", "clinical/models")
.setInputCols(Array("sentence","token","word_embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter))

val text = """Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401"""

val data = Seq(text).toDS.toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ro.med_ner.deid.subentity").predict("""
Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401""")
```

</div>

## Results

```bash
+----------------------------+---------+
|chunk                       |ner_label|
+----------------------------+---------+
|Spitalul Pentru Ochi de Deal|HOSPITAL |
|Drumul Oprea Nr.            |STREET   |
|Vaslui                      |CITY     |
|737405                      |ZIP      |
|+40(235)413773              |PHONE    |
|25 May 2022                 |DATE     |
|BUREAN MARIA                |PATIENT  |
|77                          |AGE      |
|Agota Evelyn                |DOCTOR   |
|2450502264401               |IDNUM    |
+----------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|15.1 MB|

## References

- Custom John Snow Labs datasets
- Data augmentation techniques

## Benchmarking

```bash
label      precision    recall  f1-score   support
AGE             0.96      0.99      0.97      1235
CITY            0.97      0.95      0.96       307
COUNTRY         0.92      0.74      0.82       115
DATE            0.94      0.89      0.91      5006
DOCTOR          0.96      0.96      0.96      2064
EMAIL           1.00      1.00      1.00         8
FAX             1.00      0.95      0.97        56
HOSPITAL        0.78      0.83      0.80       919
IDNUM           0.98      1.00      0.99       239
LOCATION-OTHER  1.00      0.85      0.92        13
MEDICALRECORD   1.00      1.00      1.00       455
ORGANIZATION    0.34      0.41      0.37        82
PATIENT         0.85      0.90      0.87       954
PHONE           0.97      0.98      0.98       315
PROFESSION      0.87      0.80      0.83       173
STREET          0.99      0.99      0.99       174
ZIP             0.99      0.97      0.98       140
micro-avg       0.92      0.91      0.92     12255
macro-avg       0.91      0.89      0.90     12255
weighted-avg    0.93      0.91      0.92     12255
```
