---
layout: model
title: Detect Assertion Status (assertion_jsl_augmented)
author: John Snow Labs
name: assertion_jsl_augmented
date: 2022-09-15
tags: [licensed, clinical, assertion, en]
task: Assertion Status
language: en
edition: Spark NLP for Healthcare 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The deep neural network architecture for assertion status detection in Spark NLP is based on a BiLSTM framework, and it is a modified version of the architecture proposed by Fancellu et.al. (Fancellu, Lopez, and Webber 2016). Its goal is to classify the assertions made on given medical concepts as being present, absent, or possible in the patient, conditionally present in the patient under certain circumstances, hypothetically present in the patient at some future point, and mentioned in the patient report but associated with someoneelse (Uzuner et al. 2011). This model is also the augmented version of [assertion_jsl](https://nlp.johnsnowlabs.com/2021/07/24/assertion_jsl_en.html) model with in-house annotations and it returns confidence scores of the results.

## Predicted Entities

`Present`, `Absent`, `Possible`, `Planned`, `Past`, `Family`, `Hypotetical`, `SomeoneElse`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_jsl_augmented_en_4.1.0_3.0_1663252918565.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")\

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    clinical_assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text= "The patient is here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

result = model.transform(spark.createDataFrame([[text]]).toDF('text'))
```
```scala
val documentAssembler = new DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()\
    .setInputCols(Array("document"))\
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()\
    .setInputCols(Array("sentence"))\
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(Array("sentence", "token"))\
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(Array("sentence", "token", "embeddings")) \
    .setOutputCol("ner")\

val ner_converter = new NerConverter() \
    .setInputCols(Array("sentence", "token", "ner")) \
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models") \
    .setInputCols(Array("sentence", "ner_chunk", "embeddings")) \
    .setOutputCol("assertion")
    
val nlpPipeline = Pipeline().setStages(Array(documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    clinical_assertion))

val text= Seq("The patient is here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)


```
</div>

## Results

```bash
+--------------------+-----+---+--------------------+-----------+-----------+
|           ner_chunk|begin|end|           ner_label|sentence_id|  assertion|
+--------------------+-----+---+--------------------+-----------+-----------+
|          for 2 days|   20| 29|            Duration|          0|    Present|
|          congestion|   34| 43|             Symptom|          0|    Present|
|                 mom|   47| 49|              Gender|          0|     Family|
|              yellow|   71| 76|            Modifier|          0|       Past|
|           discharge|   78| 86|             Symptom|          0|       Past|
|               nares|  107|111|External_body_par...|          0|     Absent|
|                 she|  119|121|              Gender|          0|     Family|
|                mild|  140|143|            Modifier|          0|    Present|
|problems with his...|  145|185|             Symptom|          0|    Present|
|   perioral cyanosis|  209|225|             Symptom|          0|     Absent|
|         retractions|  230|240|             Symptom|          0|     Absent|
|         One day ago|  244|254|        RelativeDate|          1|    Present|
|                 mom|  257|259|              Gender|          1|     Family|
|             Tylenol|  317|323|      Drug_BrandName|          1|       Past|
|                Baby|  326|329|                 Age|          2|       Past|
|decreased p.o. in...|  349|369|             Symptom|          2|       Past|
|                 His|  372|374|              Gender|          3|SomeoneElse|
|          20 minutes|  411|420|            Duration|          3|     Family|
|                q.2h|  422|425|           Frequency|          3|     Family|
|  to 5 to 10 minutes|  428|445|            Duration|          4|     Absent|
+--------------------+-----+---+--------------------+-----------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_jsl_augmented|
|Compatibility:|Spark NLP for Healthcare 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|6.5 MB|

## Benchmarking

```bash
       label       precision    recall  f1-score   

      Absent       0.94      0.93      0.94      
      Family       0.88      0.91      0.89       
Hypothetical       0.85      0.82      0.83       
        Past       0.89      0.89      0.89      
     Planned       0.78      0.81      0.80       
    Possible       0.82      0.82      0.82       
     Present       0.91      0.93      0.92      
 SomeoneElse       0.88      0.80      0.84       
```