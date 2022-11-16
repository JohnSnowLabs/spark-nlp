---
layout: model
title: Detect PHI for Deidentification (Subentity- Augmented)
author: John Snow Labs
name: ner_deid_subentity_augmented
date: 2021-09-03
tags: [deid, ner, en, i2b2, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.0
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN. Deidentification NER is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. It detects 23 entities. This ner model is trained with a combination of the i2b2 train set and a re-augmented version of i2b2 train set.


We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)


## Predicted Entities


`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_en_3.2.0_2.4_1630671569402.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk_subentity")

nlpPipeline = Pipeline(stages=[
                    document_assembler, 
                    sentence_detector, 
                    tokenizer, 
                    word_embeddings, 
                    deid_ner, 
                    ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""]})))
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

val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk_subentity")

val nlpPipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector, 
    tokenizer, 
    word_embeddings, 
    deid_ner, 
    ner_converter))

val data = Seq("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.deid.subentity_augmented").predict("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""")
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
|Compatibility:|Healthcare NLP 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


A custom data set which is created from the i2b2-PHI train and the re-augmented version of the i2b2-PHI train set is used.


## Benchmarking


```bash
label           tp      fp      fn      total   precision    recall      f1
PATIENT         1465.0  159.0   162.0   1627.0  0.9021      0.9004      0.9013
HOSPITAL        1417.0  120.0   167.0   1584.0  0.9219      0.8946      0.908
DATE            5513.0  57.0    129.0   5642.0  0.9898      0.9771      0.9834
ORGANIZATION    101.0   25.0    37.0    138.0   0.8016      0.7319      0.7652
CITY            277.0   47.0    64.0    341.0   0.8549      0.8123      0.8331
STREET          405.0   7.0     10.0    415.0   0.983       0.9759      0.9794
USERNAME        88.0    2.0     13.0    101.0   0.9778      0.8713      0.9215
DEVICE          10.0    0.0     0.0     10.0    1.0         1.0         1.0
IDNUM           168.0   27.0    42.0    210.0   0.8615      0.8         0.8296
STATE           172.0   15.0    33.0    205.0   0.9198      0.839       0.8776
ZIP             137.0   0.0     2.0     139.0   1.0         0.9856      0.9928
MEDICALRECORD   416.0   14.0    28.0    444.0   0.9674      0.9369      0.9519
OTHER           16.0    4.0     5.0     21.0    0.8         0.7619      0.7805
PROFESSION      261.0   22.0    75.0    336.0   0.9223      0.7768      0.8433
PHONE           328.0   21.0    20.0    348.0   0.9398      0.9425      0.9412
COUNTRY         97.0    15.0    31.0    128.0   0.8661      0.7578      0.8083
DOCTOR          3279.0  139.0   268.0   3547.0  0.9593      0.9244      0.9416
AGE             715.0   39.0    47.0    762.0   0.9483      0.9383      0.9433
macro           -       -      -        -         -           -         0.7715
micro           -       -      -        -         -           -         0.9406
```