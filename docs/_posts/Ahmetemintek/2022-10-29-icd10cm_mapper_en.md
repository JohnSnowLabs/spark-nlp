---
layout: model
title: Mapping Entities with Corresponding ICD-10-CM Codes
author: John Snow Labs
name: icd10cm_mapper
date: 2022-10-29
tags: [icd10cm, chunk_mapper, clinical, licensed, en]
task: Chunk Mapping
language: en
edition: Spark NLP for Healthcare 4.2.1
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps entities with their corresponding ICD-10-CM codes.

## Predicted Entities

`icd10cm_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_mapper_en_4.2.1_3.0_1667082016627.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol('text')\
    .setOutputCol('document')

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel\
    .pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols("sentence", "token", "ner")\
    .setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel\
    .pretrained("icd10cm_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["icd10cm_code"])

mapper_pipeline = Pipeline().setStages([
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner_model, 
    ner_converter, 
    chunkerMapper])


test_data = spark.createDataFrame([["A 35-year-old male with a history of primary leiomyosarcoma of neck, gestational diabetes mellitus diagnosed eight years prior to presentation and presented with a one-week history of polydipsia, poor appetite, and vomiting."]]).toDF("text")

mapper_model = mapper_pipeline.fit(test_data)

result= mapper_model.transform(test_data)
```
```scala
val document_assembler = new DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()\
    .setInputCols(Array("document"))\
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(Array("sentence", "token"))\
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel
    .pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols(Array("sentence", "token", "embeddings"))\
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()\
    .setInputCols("sentence", "token", "ner")\
    .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel
    .pretrained("icd10cm_mapper", "en", "clinical/models")\
    .setInputCols(Array("ner_chunk"))\
    .setOutputCol("mappings")\
    .setRels(Array("icd10cm_code")) 

val mapper_pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner_model, 
    ner_converter, 
    chunkerMapper))


val data = Seq("A 35-year-old male with a history of primary leiomyosarcoma of neck, gestational diabetes mellitus diagnosed eight years prior to presentation and presented with a one-week history of polydipsia, poor appetite, and vomiting.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data) 
```
</div>

## Results

```bash
+------------------------------+-------+------------+
|ner_chunk                     |entity |icd10cm_code|
+------------------------------+-------+------------+
|primary leiomyosarcoma of neck|PROBLEM|C49.0       |
|gestational diabetes mellitus |PROBLEM|O24.919     |
|polydipsia                    |PROBLEM|R63.1       |
|poor appetite                 |PROBLEM|R63.0       |
|vomiting                      |PROBLEM|R11.10      |
+------------------------------+-------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_mapper|
|Compatibility:|Spark NLP for Healthcare 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|6.2 MB|