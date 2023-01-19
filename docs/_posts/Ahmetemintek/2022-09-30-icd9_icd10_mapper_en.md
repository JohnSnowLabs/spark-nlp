---
layout: model
title: Mapping ICD-9-CM Codes with Their Corresponding ICD-10-CM Codes
author: John Snow Labs
name: icd9_icd10_mapper
date: 2022-09-30
tags: [en, clinical, chunk_mapping, icd9, icd10, licensed]
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

This pretrained model maps ICD-9-CM codes to corresponding ICD-10-CM codes

## Predicted Entities

`icd10_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd9_icd10_mapper_en_4.1.0_3.0_1664537323845.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd9_icd10_mapper_en_4.1.0_3.0_1664537323845.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("doc")

chunk_assembler = Doc2Chunk()\
      .setInputCols(["doc"])\
      .setOutputCol("ner_chunk")
 
chunkerMapper = ChunkMapperModel\
    .pretrained("icd9_icd10_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["icd10_code"])


mapper_pipeline = Pipeline(stages=[
    document_assembler,
    chunk_assembler,
    chunkerMapper
])


model = mapper_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("00322")
```
```scala
val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("ner_chunk")

val chunk_assembler = new Doc2Chunk()
        .setInputCols(Array("doc"))
        .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("icd9_icd10_mapper", "en","clinical/models")
        .setInputCols(Array("ner_chunk"))
        .setOutputCol("mappings")
        .setRels(Array("icd10_code"))
        
val pipeline = new Pipeline(stages = Array(
        documentAssembler,
        chunk_assembler,
        chunkerMapper))

val data = Seq("00322").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------+-------------+
|icd9_code|icd10_mapping|
+---------+-------------+
|00322    |A0222        |
+---------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd9_icd10_mapper|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|323.6 KB|
