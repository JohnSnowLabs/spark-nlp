---
layout: model
title: Mapping Entities with Corresponding ICD-9-CM Codes
author: John Snow Labs
name: icd9_mapper
date: 2022-09-30
tags: [icd9cm, chunk_mapping, en, licensed, clinical]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps entities with their corresponding ICD-9-CM codes.

## Predicted Entities

`icd9_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd9_mapper_en_4.1.0_3.0_1664535522949.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('doc')

chunk_assembler = Doc2Chunk()\
      .setInputCols(['doc'])\
      .setOutputCol('ner_chunk')
 
chunkerMapper = ChunkMapperModel\
    .pretrained("icd9_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["icd9_code"])


mapper_pipeline = Pipeline(stages=[
    document_assembler,
    chunk_assembler,
    chunkerMapper
])


test_data = spark.createDataFrame([["24 completed weeks of gestation"]]).toDF("text")

result = mapper_pipeline.fit(test_data).transform(test_data)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("doc")

val chunk_assembler = Doc2Chunk()
    .setInputCols(Array("doc"))
    .setOutputCol("ner_chunk")
 
val chunkerMapper = ChunkMapperModel
    .pretrained("icd9_mapper", "en", "clinical/models")
    .setInputCols(Array("ner_chunk"))
    .setOutputCol("mappings")
    .setRels(Array("icd9_code"))


val mapper_pipeline = new Pipeline().setStages(Array(
    document_assembler,
    chunk_assembler,
    chunkerMapper))


val test_data = Seq("24 completed weeks of gestation").toDS.toDF("text")

val result = mapper_pipeline.fit(test_data).transform(test_data) 
```
</div>

## Results

```bash
+-------------------------------+------------+
|chunk                          |icd9_mapping|
+-------------------------------+------------+
|24 completed weeks of gestation|765.22      |
+-------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd9_mapper|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|374.4 KB|