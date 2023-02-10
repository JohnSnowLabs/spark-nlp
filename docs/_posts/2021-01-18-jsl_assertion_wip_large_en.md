---
layout: model
title: Detect Assertion Status (assertion_wip_large)
author: John Snow Labs
name: jsl_assertion_wip_large
date: 2021-01-18
task: Assertion Status
language: en
edition: Healthcare NLP 2.7.0
spark_version: 2.4
tags: [clinical, licensed, assertion, en, ner]
supported: true
article_header:
    type: cover
use_language_switcher: "Python-Scala-Java"
---
 
## Description


The deep neural network architecture for assertion status detection in Spark NLP is based on a BiLSTM framework, and is a modified version of the architecture proposed by Fancellu et.al. (Fancellu, Lopez, and Webber 2016). Its goal is to classify the assertions made on given medical concepts as being present, absent, or possible in the patient, conditionally present in the patient under certain circumstances, hypothetically present in the patient at some future point, and mentioned in the patient report but associated with someone- else (Uzuner et al. 2011). Apart from what we released in other assertion models, an in-house annotations on a curated dataset (6K clinical notes) is used to augment the base assertion dataset (2010 i2b2/VA).


{:.h2_title}
## Predicted Entities
`present`, `absent`, `possible`, `planned`, `someoneelse`, `past`.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_assertion_wip_large_en_2.6.5_2.4_1609091911183.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_assertion_wip_large_en_2.6.5_2.4_1609091911183.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


{:.h2_title}
## How to use


Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionDLModel.


<div class="tabs-box" markdown="1">


{% include programmingLanguageSelectScalaPython.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("jsl_assertion_wip_large", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])

light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```


```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val nerConverter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("jsl_assertion_wip_large", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))

val data = Seq("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


</div>


{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an ``"assertion"`` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select ``"ner_chunk.result"`` and ``"assertion.result"`` from your output dataframe.


```bash
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
|Model Name:|jsl_assertion_wip_large|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|[en]|
|Case sensitive:|false|


{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/


{:.h2_title}
## Benchmarking
```bash
label           prec   rec    f1   
absent          0.957  0.949  0.953
someoneelse     0.958  0.936  0.947
planned         0.766  0.657  0.707
possible        0.852  0.884  0.868
past            0.894  0.890  0.892
present         0.902  0.917  0.910
Macro-average   0.888  0.872  0.880
Micro-average   0.908  0.908  0.908
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbODc0MDA5NzYyXX0=
-->