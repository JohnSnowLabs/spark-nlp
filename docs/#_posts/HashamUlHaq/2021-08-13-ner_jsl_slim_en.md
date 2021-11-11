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

`Death_Entity`, `Medical_Device`, `Vital_Sign`, `Alergen`, `Drug`, `Clinical_Dept`, `Lifestyle`, `Symptom`, `Body_Part`, `Physical_Measurement`, `Admission_Discharge`, `Date_Time`, `Age`, `Birth_Entity`, `Header`, `Oncological`, `Substance_Quantity`, `Test_Result`, `Test`, `Procedure`, `Treatment`, `Disease_Syndrome_Disorder`, `Pregnancy_Newborn`, `Demographics`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL/){:.button.button-orange}
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

results = model.transform(spark.createDataFrame([["HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer."]], ["text"]))
```
```scala
...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(["sentence", "token"])
   .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models") 
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val data = Seq("HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk            | entity       |
|---:|:-----------------|:-------------|
|  0 | HISTORY:         | Header       |
|  1 | 30-year-old      | Age          |
|  2 | female           | Demographics |
|  3 | mammography      | Test         |
|  4 | soft tissue lump | Symptom      |
|  5 | shoulder         | Body_Part    |
|  6 | breast cancer    | Oncological  |
|  7 | her mother       | Demographics |
|  8 | age 58           | Age          |
|  9 | breast cancer    | Oncological  |

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
B-Medical_Device	 2696	 444	 282	 0.8585987	 0.90530556	 0.88133377
I-Physical_Measurement	 220	 16	 34	 0.9322034	 0.86614174	 0.8979592
B-Alergen	 0	 0	 6	 0.0	 0.0	 0.0
B-Procedure	 1800	 239	 281	 0.8827857	 0.8649688	 0.8737864
I-Substance_Quantity	 0	 0	 6	 0.0	 0.0	 0.0
B-Drug	 1865	 218	 237	 0.89534324	 0.88725024	 0.8912784
I-Test_Result	 289	 203	 292	 0.58739835	 0.49741825	 0.5386766
I-Pregnancy_Newborn	 150	 41	 104	 0.7853403	 0.5905512	 0.6741573
B-Substance_Quantity	 0	 0	 2	 0.0	 0.0	 0.0
B-Admission_Discharge	 255	 35	 6	 0.87931037	 0.9770115	 0.92558986
B-Demographics	 4609	 119	 123	 0.9748308	 0.9740068	 0.97441864
I-Admission_Discharge	 0	 0	 3	 0.0	 0.0	 0.0
I-Lifestyle	 71	 49	 20	 0.59166664	 0.7802198	 0.67298573
B-Header	 2463	 53	 122	 0.9789348	 0.9528046	 0.965693
I-Date_Time	 928	 184	 191	 0.8345324	 0.8293119	 0.83191395
B-Test_Result	 866	 198	 262	 0.81390977	 0.7677305	 0.79014593
I-Treatment	 114	 37	 46	 0.7549669	 0.7125	 0.733119
B-Clinical_Dept	 688	 83	 76	 0.8923476	 0.90052354	 0.8964169
B-Test	 1920	 333	 313	 0.85219705	 0.85982984	 0.8559965
B-Death_Entity	 36	 9	 2	 0.8	 0.94736844	 0.8674699
B-Lifestyle	 268	 58	 50	 0.8220859	 0.8427673	 0.8322981
B-Date_Time	 823	 154	 176	 0.8423746	 0.8238238	 0.83299595
I-Age	 136	 34	 49	 0.8	 0.73513514	 0.7661972
I-Oncological	 345	 41	 19	 0.8937824	 0.9478022	 0.91999996
I-Body_Part	 3717	 720	 424	 0.8377282	 0.8976093	 0.8666356
B-Pregnancy_Newborn	 153	 51	 104	 0.75	 0.5953307	 0.6637744
B-Treatment	 169	 41	 58	 0.8047619	 0.74449337	 0.7734553
I-Procedure	 2302	 326	 417	 0.8759513	 0.8466348	 0.8610435
B-Birth_Entity	 6	 5	 7	 0.54545456	 0.46153846	 0.5
I-Vital_Sign	 639	 197	 93	 0.76435405	 0.8729508	 0.815051
I-Header	 4451	 111	 216	 0.97566855	 0.9537176	 0.9645682
I-Death_Entity	 2	 0	 0	 1.0	 1.0	 1.0
I-Clinical_Dept	 621	 54	 39	 0.92	 0.9409091	 0.9303371
I-Test	 1593	 378	 353	 0.8082192	 0.81860226	 0.81337756
B-Age	 472	 43	 51	 0.91650486	 0.90248567	 0.9094413
I-Symptom	 4227	 1271	 1303	 0.76882505	 0.7643761	 0.7665941
I-Demographics	 321	 53	 53	 0.85828876	 0.85828876	 0.85828876
B-Body_Part	 6312	 912	 809	 0.87375414	 0.88639235	 0.8800279
B-Physical_Measurement	 91	 10	 17	 0.9009901	 0.8425926	 0.8708134
B-Disease_Syndrome_Disorder	 2817	 336	 433	 0.8934348	 0.86676925	 0.8799001
B-Symptom	 4522	 830	 747	 0.8449178	 0.8582274	 0.8515206
I-Disease_Syndrome_Disorder	 2814	 386	 530	 0.879375	 0.8415072	 0.86002445
I-Drug	 3737	 612	 517	 0.859278	 0.8784673	 0.8687667
I-Medical_Device	 1825	 331	 131	 0.84647495	 0.9330266	 0.8876459
B-Oncological	 276	 28	 27	 0.90789473	 0.9108911	 0.9093904
B-Vital_Sign	 429	 97	 79	 0.81558937	 0.8444882	 0.8297872
tp: 62038 fp: 9340 fn: 9110 labels: 46
Macro-average	 prec: 0.76782775, rec: 0.7648211, f1: 0.76632154
Micro-average	 prec: 0.86914736, rec: 0.87195706, f1: 0.87055

```
