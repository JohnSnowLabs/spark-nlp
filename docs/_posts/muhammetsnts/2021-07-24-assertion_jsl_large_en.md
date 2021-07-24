---
layout: model
title: Detect Assertion Status (assertion_jsl_large)
author: John Snow Labs
name: assertion_jsl_large
date: 2021-07-24
tags: [licensed, clinical, assertion, en]
task: Assertion Status
language: en
edition: Spark NLP for Healthcare 3.1.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deep learning named entity recognition model for assertions. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities

`present`, `absent`, `possible`, `planned`, `someoneelse`, `past`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_jsl_large_en_3.1.2_2.4_1627156678782.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_large", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

result = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"])
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings")) 
  .setOutputCol("ner")

val nerConverter = NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_large", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))

val data = Seq("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
The output is a dataframe with a sentence per row and an `assertion` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select `ner_chunk.result` and `assertion.result` from your output dataframe.

+-----------------------------------------+-----+---+----------------------------+-------+-----------+
|chunk                                    |begin|end|ner_label                   |sent_id|assertion  |
+-----------------------------------------+-----+---+----------------------------+-------+-----------+
|21-day-old                               |17   |26 |Age                         |0      |present    |
|Caucasian                                |28   |36 |Race_Ethnicity              |0      |present    |
|male                                     |38   |41 |Gender                      |0      |someoneelse|
|for 2 days                               |48   |57 |Duration                    |0      |present    |
|congestion                               |62   |71 |Symptom                     |0      |present    |
|mom                                      |75   |77 |Gender                      |0      |someoneelse|
|yellow                                   |99   |104|Modifier                    |0      |present    |
|discharge                                |106  |114|Symptom                     |0      |present    |
|nares                                    |135  |139|External_body_part_or_region|0      |someoneelse|
|she                                      |147  |149|Gender                      |0      |present    |
|mild                                     |168  |171|Modifier                    |0      |present    |
|problems with his breathing while feeding|173  |213|Symptom                     |0      |present    |
|perioral cyanosis                        |237  |253|Symptom                     |0      |absent     |
|retractions                              |258  |268|Symptom                     |0      |absent     |
|One day ago                              |272  |282|RelativeDate                |1      |someoneelse|
|mom                                      |285  |287|Gender                      |1      |someoneelse|
|Tylenol                                  |345  |351|Drug_BrandName              |1      |someoneelse|
|Baby                                     |354  |357|Age                         |2      |someoneelse|
|decreased p.o. intake                    |377  |397|Symptom                     |2      |someoneelse|
|His                                      |400  |402|Gender                      |3      |someoneelse|
+-----------------------------------------+-----+---+----------------------------+-------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_jsl_large|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with ‘embeddings_clinical’. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking

```bash
|   | label           | prec  | rec   | f1    |
|--:|----------------:|------:|------:|------:|
| 0 | absent           | 0.957 | 0.949 | 0.953 |
| 1 | someoneelse      | 0.958 | 0.936 | 0.947 |
| 2 | planned          | 0.766 | 0.657 | 0.707 |
| 3 | possible         | 0.852 | 0.884 | 0.868 |
| 4 | past             | 0.894 | 0.890 | 0.892 |
| 5 | present          | 0.902 | 0.917 | 0.910 |
| 6 | Macro-average    | 0.888 | 0.872 | 0.880 |
| 7 | Micro-average    | 0.908 | 0.908 | 0.908 |
```
