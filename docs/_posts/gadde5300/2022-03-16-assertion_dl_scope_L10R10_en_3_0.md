---
layout: model
title: Detect Assertion Status (assertion_dl_scope_L10R10)
author: John Snow Labs
name: assertion_dl_scope_L10R10
date: 2022-03-16
tags: [clinical, licensed, en, assertion]
task: Assertion Status
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model considers 10 tokens on the left and 10 tokens on the right side of the clinical entities extracted by NER models and assigns their assertion status based on their context in this scope.

## Predicted Entities

`present`, `absent`, `possible`, `conditional`, `associated_with_someone_else`, `hypothetical`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_scope_L10R10_en_3.4.1_3.0_1647431567296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

token = Tokenizer()\
    .setInputCols(['sentence'])\
    .setOutputCol('token')

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl_scope_L10R10", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setScopeWindow([10,10])\
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[document,sentenceDetector, token, word_embeddings,clinical_ner,ner_converter,  clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_result = LightPipeline(model).fullAnnotate("She has no history of liver disease , hepatitis .")[0]

```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
val sentenceDetector = new SentenceDetector()\
  .setInputCols(Array("document"))
  .setOutputCol("sentence")
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings")) 
  .setOutputCol("ner")
val ner_converter = NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
val clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setScopeWindow(Array(10,10))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))

val data = Seq("She has no history of liver disease , hepatitis .").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+---------+-------+
|chunk        |assertion|entity |
+-------------+---------+-------+
|liver disease|absent   |PROBLEM|
|hepatitis    |absent   |PROBLEM|
+-------------+---------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_scope_L10R10|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|1.4 MB|

## References

Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with ‘embeddings_clinical’. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking

```bash
Total test loss: 35.6458	                  
Avg test loss: 0.2875

|label                       |tp  |fp |fn |prec      |rec      |f1        |
|----------------------------|----|---|---|----------|---------|----------|
|absent                      |812 |48 |71 |0.94418603|0.9195923|0.93172693|
|present                     |2463|127|141|0.9509652 |0.9458525|0.948402  |
|conditional                 |25  |19 |28 |0.5681818 |0.4716981|0.5154639 |
|associated_with_someone_else|36  |7  |9  |0.8372093 |0.8      |0.8181818 |
|hypothetical                |147 |31 |28 |0.8258427 |0.84     |0.8328612 |
|possible                    |159 |87 |42 |0.64634144|0.7910448|0.71140933|

tp: 3642                  fp: 319                     fn: 319                  labels: 6
Macro-average	 prec: 0.79545444, rec: 0.79469794, f1: 0.795076
Micro-average	 prec: 0.91946477, rec: 0.91946477, f1: 0.91946477
```
