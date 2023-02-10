---
layout: model
title: Detect PHI for deidentification purposes
author: John Snow Labs
name: ner_deid_subentity_augmented_i2b2
date: 2021-11-29
tags: [deid, ner, phi, deidentification, licensed, i2b2, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition model that finds Protected Health Information (PHI) for deidentification purposes. 
This NER model is trained with a reviewed version of the re-augmented 2014 i2b2 Deid dataset, and detects up to 23 entity types.

We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)

## Predicted Entities

`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_i2b2_en_3.3.2_2.4_1638185564971.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_i2b2_en_3.3.2_2.4_1638185564971.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentence_detector = SentenceDetector() \
      .setInputCols(["document"]) \
      .setOutputCol("sentence")

tokenizer = Tokenizer() \
      .setInputCols(["sentence"]) \
      .setOutputCol("token")

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

val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented_i2b2", "en", "clinical/models") 
      .setInputCols(Array("sentence", "token", "embeddings")) 
      .setOutputCol("ner")

val ner_converter = NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk_subentity")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))

val result = nlpPipeline.fit(Seq("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 years old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""").toDS.toDF("text")).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.deid_subentity_augmented_i2b2").predict("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 years old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""")
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
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

In-house annotations based on `2014 i2b2 Deid dataset`.

## Benchmarking

(on official test set from 2014 i2b2 Deid Data-set)

```bash
label  precision    recall  f1-score   support
AGE       0.96      0.96      0.96       764
CITY       0.83      0.84      0.84       260
COUNTRY       0.79      0.85      0.82       117
DATE       0.97      0.97      0.97      4980
DEVICE       0.88      0.88      0.88         8
DOCTOR       0.94      0.88      0.91      1912
HOSPITAL       0.91      0.83      0.87       875
IDNUM       0.84      0.85      0.84       195
LOCATION-OTHER       0.86      0.46      0.60        13
MEDICALRECORD       0.98      0.95      0.96       422
ORGANIZATION       0.83      0.59      0.69        82
PATIENT       0.93      0.93      0.93       879
PHONE       0.93      0.91      0.92       215
PROFESSION       0.84      0.75      0.79       179
STATE       0.95      0.86      0.90       190
STREET       0.96      0.97      0.97       136
USERNAME       1.00      0.96      0.98        92
ZIP       0.98      0.99      0.98       140
micro-avg       0.95      0.92      0.94     11459
macro-avg       0.86      0.81      0.83     11459
weighted-avg       0.95      0.92      0.93     11459
```
`FAX` and `EMAIL` has been removed from official i2b2 test-set since there is not enough data to train in the official i2b2 train-set.
