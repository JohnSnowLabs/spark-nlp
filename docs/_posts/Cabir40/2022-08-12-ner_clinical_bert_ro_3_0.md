---
layout: model
title: Detect Clinical Entities in Romanian (Bert, Base, Cased)
author: John Snow Labs
name: ner_clinical_bert
date: 2022-08-12
tags: [licensed, clinical, ro, ner, bert]
task: Named Entity Recognition
language: ro
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract clinical entities from Romanian clinical texts. This model is trained using `bert_base_cased` embeddings.

## Predicted Entities

`Measurements`, `Form`, `Symptom`, `Route`, `Procedure`, `Disease_Syndrome_Disorder`, `Score`, `Drug_Ingredient`, `Pulse`, `Frequency`, `Date`, `Body_Part`, `Drug_Brand_Name`, `Time`, `Direction`, `Medical_Device`, `Imaging_Technique`, `Test`, `Imaging_Findings`, `Imaging_Test`, `Test_Result`, `Weight`, `Clinical_Dept`, `Units`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_bert_ro_4.0.2_3.0_1660295520992.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

word_embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_bert","ro","clinical/models")\
.setInputCols(["sentence","token","embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence","token","ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter])

data = spark.createDataFrame([[""" Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Scout. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min."""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)

```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_clinical_bert", "ro", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new PipelineModel().setStages(Array(document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter))

val data = Seq("""Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Scout. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("ro.embed.clinical.bert.base_cased").predict(""" Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Scout. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min.""")
```
</div>

## Results

```bash
+--------------------------+-------------------------+
|chunks                    |entities                 |
+--------------------------+-------------------------+
|Angio CT cardio-toracic   |Imaging_Test             |
|Atrezie                   |Disease_Syndrome_Disorder|
|valva pulmonara           |Body_Part                |
|Hipoplazie                |Disease_Syndrome_Disorder|
|VS                        |Body_Part                |
|Atrezie                   |Disease_Syndrome_Disorder|
|VAV stang                 |Body_Part                |
|Anastomoza Glenn          |Disease_Syndrome_Disorder|
|Tromboza                  |Disease_Syndrome_Disorder|
|Sectia Clinica Cardiologie|Clinical_Dept            |
|GE Revolution HD          |Medical_Device           |
|Branula albastra          |Medical_Device           |
|membrului superior drept  |Body_Part                |
|Scout                     |Body_Part                |
|30 ml                     |Dosage                   |
|Iomeron 350               |Drug_Ingredient          |
|2.2 ml/s                  |Dosage                   |
|20 ml                     |Dosage                   |
|ser fiziologic            |Drug_Ingredient          |
|angio-CT                  |Imaging_Test             |
+--------------------------+-------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_bert|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|16.2 MB|

## Benchmarking

```bash
label                  precision    recall  f1-score   support
Body_Part                   0.91      0.93      0.92       679
Clinical_Dept               0.68      0.65      0.67        97
Date                        0.99      0.99      0.99        87
Direction                   0.66      0.76      0.70        50
Disease_Syndrome_Disorder   0.73      0.76      0.74       121
Dosage                      0.78      1.00      0.87        38
Drug_Ingredient             0.90      0.94      0.92        48
Form                        1.00      1.00      1.00         6
Imaging_Findings            0.86      0.82      0.84       201
Imaging_Technique           0.92      0.92      0.92        26
Imaging_Test                0.93      0.98      0.95       205
Measurements                0.71      0.69      0.70       214
Medical_Device              0.85      0.81      0.83        42
Pulse                       0.82      1.00      0.90         9
Route                       1.00      0.91      0.95        33
Score                       1.00      0.98      0.99        41
Time                        1.00      1.00      1.00        28
Units                       0.60      0.93      0.73        88
Weight                      0.82      1.00      0.90         9
micro-avg                   0.84      0.87      0.86      2037
macro-avg                   0.70      0.74      0.72      2037
weighted-avg                0.84      0.87      0.85      2037
```
