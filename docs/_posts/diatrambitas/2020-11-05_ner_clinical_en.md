---
layout: model
title: Ner DL Model Clinical
author: John Snow Labs
name: ner_clinical
date: 2020-11-05
tags: [en, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Pretrained named entity recognition deep learning model for clinical terms. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities
``PROBLEM``, ``TEST``, ``TREATMENT``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_SIGN_SYMP/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_en_2.6.2_2.4_1604593036784.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use


<div class="tabs-box" markdown="1">
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."""]})))

```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val result = pipeline.fit(Seq.empty["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."].toDS.toDF("text")).transform(data)
```
</div>

## Results
{:.result_box}
```bash
+-------------------------------------+---------+
|chunk                                |ner      |
+-------------------------------------+---------+
|congestion                           |PROBLEM  |
|suctioning yellow discharge          |PROBLEM  |
|some mild problems with his breathing|PROBLEM  |
|any perioral cyanosis                |PROBLEM  |
|retractions                          |PROBLEM  |
|a tactile temperature                |TEST     |
|Tylenol                              |TREATMENT|
|his respiratory congestion           |PROBLEM  |
|more tired                           |PROBLEM  |
|fussy                                |PROBLEM  |
|albuterol treatments                 |TREATMENT|
|His urine output                     |TEST     |
|dirty diapers                        |TREATMENT|
|diarrhea                             |PROBLEM  |
|yellow colored                       |PROBLEM  |
+-------------------------------------+---------+
```

## Model Parameters
{:.table-model}
|---|---|
|Model Name:|ner_clinical|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source
Trained on 2010 i2b2 challenge data with embeddings_clinical. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking
|    | label         |     tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:--------------|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | I-TREATMENT   |   6492 |   873 | 1445 | 0.881466 | 0.817941 | 0.848517 |
|  1 | I-PROBLEM     |  15645 |  1808 | 2031 | 0.896408 | 0.885098 | 0.890717 |
|  2 | B-PROBLEM     |  11160 |  1048 | 1424 | 0.914155 | 0.88684  | 0.90029  |
|  3 | I-TEST        |   6878 |   864 | 1132 | 0.888401 | 0.858677 | 0.873286 |
|  4 | B-TEST        |   8140 |   932 | 1081 | 0.897266 | 0.882768 | 0.889958 |
|  5 | B-TREATMENT   |   8163 |   945 | 1150 | 0.896245 | 0.876517 | 0.886271 |
|  6 | Macro-average | 56478  | 6470  | 8263 | 0.895657 | 0.867974 | 0.881598 |
|  7 | Micro-average | 56478  | 6470  | 8263 | 0.897217 | 0.872368 | 0.884618 |