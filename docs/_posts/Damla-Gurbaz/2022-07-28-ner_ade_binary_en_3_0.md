---
layout: model
title: Detect Binary Adverse Drug Events
author: John Snow Labs
name: ner_ade_binary
date: 2022-07-28
tags: [clinical, ner, ade, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect binary adverse reactions of drugs in reviews, tweets, and medical text using pretrained NER model.

## Predicted Entities

`ADE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_binary_en_4.0.0_3.0_1658993120506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
 
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_ade_binary", "en", "clinical/models")
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner") 

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        clinical_embeddings,
        clinical_ner,
        ner_converter])


model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["I used to be on paxil but that made me more depressed and prozac made me angry",
                              "Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs."], StringType()).toDF("text")
 
    
result = model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
        
val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_ade_binary", "en", "clinical/models")
    .setInputCols(Array("sentence","token","embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","ner"))
    .setOutputCol("ner_chunk")

val nlpPipeline = new Pipeline(stages=Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        clinical_embeddings,
        clinical_ner,
        ner_converter))


val data = Seq(Array("I used to be on paxil but that made me more depressed and prozac made me angry",
                     "Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs.")).toDS().toDF("text")
    
val result = nlpPipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+---------+
|chunk        |ner_label|
+-------------+---------+
|depressed    |ADE      |
|angry        |ADE      |
|sugar crashes|ADE      |
+-------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ade_binary|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.1 MB|

## Benchmarking

```bash
    label   tp    fp    fn    prec    rec    f1
    I-ADE   1383  417   304   0.768   0.819  0.793
    B-ADE   1214  311   238   0.796   0.836  0.815
Macro-avg   2597  728   542   0.782   0.827  0.804 
Micro-avg   2597  728   542   0.781   0.827  0.803
```
