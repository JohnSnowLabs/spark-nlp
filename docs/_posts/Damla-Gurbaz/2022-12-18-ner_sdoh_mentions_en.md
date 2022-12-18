---
layout: model
title: Detect Social Determinants of Health Mentions
author: John Snow Labs
name: ner_sdoh_mentions
date: 2022-12-18
tags: [en, licensed, ner, sdoh, mentions, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model is intended for detecting Social Determinants of Health mentions in clinical notes and trained by using MedicalNerApproach annotator that allows to train generic NER models based on Neural Networks.

## Predicted Entities

`sdoh_community`, `sdoh_economics`, `sdoh_education`, `sdoh_environment`, `behavior_tobacco`, `behavior_alcohol`, `behavior_drug`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_mentions_en_4.2.2_3.0_1671369830893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")\
      
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols("document")\
    .setOutputCol("sentence")
    
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")
    
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")
    
ner_model = MedicalNerModel.pretrained("ner_sdoh_mentions", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")
    
ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[
    document_assembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter])

df = spark.createDataFrame([["Mr. Known lastname 9880 is a pleasant, cooperative gentleman with a long standing history (20 years) diverticulitis. He is married and has 3 children. He works in a bank. He denies any alcohol or intravenous drug use. He has been smoking for many years."]]).toDF("text")

result = nlpPipeline.fit(df).transform(df)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")
    
val ner_model = MedicalNerModel.pretrained("ner_sdoh_mentions", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")
    
val nlpPipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                sentenceDetector,
                                                tokenizer,
                                                embeddings,
                                                ner_model,
                                                ner_converter))

val data = Seq("Mr. Known lastname 9880 is a pleasant, cooperative gentleman with a long standing history (20 years) diverticulitis. He is married and has 3 children. He works in a bank. He denies any alcohol or intravenous drug use. He has been smoking for many years.").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------+----------------+
|chunk           |ner_label       |
+----------------+----------------+
|married         |sdoh_community  |
|children        |sdoh_community  |
|works           |sdoh_economics  |
|alcohol         |behavior_alcohol|
|intravenous drug|behavior_drug   |
|smoking         |behavior_tobacco|
+----------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_mentions|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.1 MB|

## Benchmarking

```bash
           label  precision    recall  f1-score   support
behavior_alcohol       0.95      0.94      0.94       798
   behavior_drug       0.93      0.92      0.92       366
behavior_tobacco       0.95      0.95      0.95       936
  sdoh_community       0.97      0.97      0.97       969
  sdoh_economics       0.95      0.91      0.93       363
  sdoh_education       0.69      0.65      0.67        34
sdoh_environment       0.93      0.90      0.92       651
       micro avg       0.95      0.94      0.94      4117
       macro avg       0.91      0.89      0.90      4117
    weighted avg       0.95      0.94      0.94      4117
```
