---
layout: model
title: Detect Assertion Status (assertion_jsl_augmented)
author: John Snow Labs
name: assertion_jsl_augmented
date: 2022-09-15
tags: [licensed, clinical, assertion, en]
task: Assertion Status
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_jsl_augmented_en_4.1.0_3.0_1663252918565.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_jsl_augmented_en_4.1.0_3.0_1663252918565.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setBlackList(["RelativeDate", "Gender"])

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


text = """Patient had a headache for the last 2 weeks, and appears anxious when she walks fast. No alopecia noted. She denies pain. Her father is paralyzed and it is a stressor for her. She was bullied by her boss and got antidepressant. We prescribed sleeping pills for her current insomnia"""

data = spark.createDataFrame([[text]]).toDF('text')

result = nlpPipeline.fit(data).transform(data)

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

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk") 
    .setBlackList(Array("RelativeDate", "Gender"))
    
val clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models") 
    .setInputCols(Array("sentence", "ner_chunk", "embeddings")) 
    .setOutputCol("assertion")
    
val nlpPipeline = Pipeline().setStages(Array(documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    clinical_assertion))

val data= Seq("Patient had a headache for the last 2 weeks, and appears anxious when she walks fast. No alopecia noted. She denies pain. Her father is paralyzed and it is a stressor for her. She was bullied by her boss and got antidepressant. We prescribed sleeping pills for her current insomnia").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)


```
</div>

## Results

```bash
+--------------+-----+---+-------------------------+-----------+---------+
|ner_chunk     |begin|end|ner_label                |sentence_id|assertion|
+--------------+-----+---+-------------------------+-----------+---------+
|headache      |14   |21 |Symptom                  |0          |Past     |
|anxious       |57   |63 |Symptom                  |0          |Possible |
|alopecia      |89   |96 |Disease_Syndrome_Disorder|1          |Absent   |
|pain          |116  |119|Symptom                  |2          |Absent   |
|paralyzed     |136  |144|Symptom                  |3          |Family   |
|antidepressant|212  |225|Drug_Ingredient          |4          |Past     |
|sleeping pills|242  |255|Drug_Ingredient          |5          |Planned  |
|insomnia      |273  |280|Symptom                  |5          |Present  |
+--------------+-----+---+-------------------------+-----------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_jsl_augmented|
|Compatibility:|Healthcare NLP 4.1.0+|
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
