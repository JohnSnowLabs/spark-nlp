---
layout: model
title: Detect Clinical Entities (ner_jsl_slim)
author: John Snow Labs
name: ner_jsl_slim
date: 2021-08-13
tags: [ner, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.2.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a pretrained named entity recognition deep learning model for clinical terminology. It is based on `ner_jsl` model, but with more generalised entities.

## Predicted Entities

`Body_Part`, `Admission_Discharge`, `Clinical_Dept`, `Medical_Device`, `Procedure`, `Substance_Quantity`, `Drug`, `Disease_Syndrome_Disorder`, `Alergen`, `Pregnancy_Newborn`, `Symptom`, `Header`, `Vital_Sign`, `Test`, `Lifestyle`, `Birth_Entity`, `Demographics`, `Death_Entity`, `Age`, `Test_Result`, `Physical_Measurement`, `Treatment`, `Date_Time`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_SIGN_SYMP/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_slim_en_3.2.0_3.0_1628875762291.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings_clinical = WordEmbeddingsModel().pretrained('embeddings_clinical', 'en', 'clinical/models') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

clinical_ner = MedicalNerModel().pretrained("ner_jsl_slim", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```
```scala
...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(["sentence", "token"])
   .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_jsl_greedy_biobert", "en", "clinical/models") 
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val data = Seq("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk                                          | entity        |
|---:|:-----------------------------------------------|:--------------|
|  0 | 21-day-old                                     | Age           |
|  1 | Caucasian male                                 | Demographics  |
|  2 | congestion                                     | Symptom       |
|  3 | mom                                            | Demographics  |
|  4 | yellow discharge                               | Symptom       |
|  5 | nares                                          | Body_Part     |
|  6 | she                                            | Demographics  |
|  7 | mild problems with his breathing while feeding | Symptom       |
|  8 | perioral cyanosis                              | Symptom       |
|  9 | retractions                                    | Symptom       |
| 10 | One day ago                                    | Date_Time     |
| 11 | mom                                            | Demographics  |
| 12 | tactile temperature                            | Symptom       |
| 13 | Tylenol                                        | Drug          |
| 14 | Baby                                           | Age           |
| 15 | decreased p.o. intake                          | Symptom       |
| 16 | His                                            | Demographics  |
| 17 | his                                            | Demographics  |
| 18 | respiratory congestion                         | Symptom       |
| 19 | He                                             | Demographics  |
| 20 | tired                                          | Symptom       |
| 21 | fussy                                          | Symptom       |
| 22 | over the past 2 days                           | Date_Time     |
| 23 | albuterol                                      | Drug          |
| 24 | ER                                             | Clinical_Dept |
| 25 | His                                            | Demographics  |
| 26 | urine output has also decreased                | Symptom       |
| 27 | he                                             | Demographics  |
| 28 | he                                             | Demographics  |
| 29 | Mom                                            | Demographics  |
| 30 | diarrhea                                       | Symptom       |
| 31 | His                                            | Demographics  |
| 32 | bowel                                          | Body_Part     |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_slim|
|Compatibility:|Spark NLP for Healthcare 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on data annotated by JSL.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-Medical_Device	 1695	 340	 227	 0.8329238	 0.8818939	 0.8567096
I-Physical_Measurement	 163	 44	 12	 0.7874396	 0.93142855	 0.85340315
B-Alergen	 1	 0	 8	 1.0	 0.11111111	 0.19999999
B-Procedure	 1337	 209	 159	 0.86481243	 0.8937166	 0.87902695
I-Substance_Quantity	 2	 2	 6	 0.5	 0.25	 0.33333334
B-Drug	 1701	 226	 155	 0.8827193	 0.9164871	 0.8992863
I-Test_Result	 965	 116	 256	 0.89269197	 0.7903358	 0.8384014
I-Pregnancy_Newborn	 81	 27	 54	 0.75	 0.6	 0.6666667
B-Substance_Quantity	 1	 0	 2	 1.0	 0.33333334	 0.5
B-Admission_Discharge	 186	 8	 8	 0.9587629	 0.9587629	 0.9587629
B-Demographics	 2660	 73	 77	 0.97328943	 0.971867	 0.9725777
I-Admission_Discharge	 0	 0	 2	 0.0	 0.0	 0.0
I-Lifestyle	 65	 21	 31	 0.75581396	 0.6770833	 0.7142857
B-Header	 1882	 117	 88	 0.94147074	 0.95532995	 0.94834965
I-Date_Time	 878	 155	 94	 0.8499516	 0.9032922	 0.8758105
B-Test_Result	 982	 124	 273	 0.88788426	 0.7824701	 0.83185095
I-Treatment	 108	 39	 103	 0.7346939	 0.51184833	 0.60335195
B-Clinical_Dept	 520	 73	 67	 0.87689716	 0.8858603	 0.8813559
B-Test	 1583	 288	 292	 0.8460716	 0.84426665	 0.8451682
B-Death_Entity	 21	 5	 3	 0.8076923	 0.875	 0.84
B-Lifestyle	 190	 28	 48	 0.8715596	 0.79831934	 0.8333333
B-Date_Time	 882	 126	 135	 0.875	 0.86725664	 0.8711111
I-Age	 130	 15	 26	 0.8965517	 0.8333333	 0.8637873
I-Body_Part	 4079	 437	 436	 0.90323293	 0.903433	 0.903333
B-Pregnancy_Newborn	 62	 26	 55	 0.70454544	 0.52991456	 0.60487807
B-Treatment	 146	 48	 87	 0.7525773	 0.62660944	 0.68384075
I-Procedure	 1719	 251	 195	 0.8725888	 0.89811915	 0.8851699
B-Birth_Entity	 2	 1	 5	 0.6666667	 0.2857143	 0.4
I-Vital_Sign	 466	 177	 62	 0.7247278	 0.88257575	 0.79590094
I-Header	 1475	 79	 81	 0.94916344	 0.94794345	 0.9485531
I-Death_Entity	 0	 0	 1	 0.0	 0.0	 0.0
I-Clinical_Dept	 377	 37	 33	 0.910628	 0.9195122	 0.91504854
I-Test	 1258	 299	 286	 0.807964	 0.8147668	 0.8113512
B-Age	 318	 25	 40	 0.9271137	 0.8882682	 0.9072753
I-Symptom	 2719	 756	 990	 0.782446	 0.7330817	 0.75695986
I-Demographics	 167	 12	 21	 0.93296087	 0.88829786	 0.9100817
B-Body_Part	 3962	 427	 419	 0.90271133	 0.90435976	 0.90353477
B-Physical_Measurement	 62	 15	 21	 0.8051948	 0.74698794	 0.77500004
B-Disease_Syndrome_Disorder	 2301	 299	 261	 0.885	 0.8981265	 0.89151496
B-Symptom	 3047	 546	 548	 0.84803784	 0.84756607	 0.8478019
I-Disease_Syndrome_Disorder	 2216	 288	 287	 0.884984	 0.8853376	 0.88516074
I-Drug	 2018	 264	 238	 0.88431203	 0.89450353	 0.8893786
I-Medical_Device	 1130	 209	 122	 0.8439134	 0.9025559	 0.8722501
B-Vital_Sign	 355	 103	 96	 0.7751092	 0.7871397	 0.7810781
I-Alergen	 1	 0	 1	 1.0	 0.5	 0.6666667
tp: 43913 fp: 6335 fn: 6411 labels: 45
Macro-average	 prec: 0.8122245, rec: 0.7390624, f1: 0.7739183
Micro-average	 prec: 0.8739253, rec: 0.8726055, f1: 0.87326497

```