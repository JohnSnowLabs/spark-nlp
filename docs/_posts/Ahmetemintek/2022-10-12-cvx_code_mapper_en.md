---
layout: model
title: Mapping CVX Codes with Their Corresponding Vaccine Names and CPT Codes.
author: John Snow Labs
name: cvx_code_mapper
date: 2022-10-12
tags: [cvx, cpt, chunk_mapping, en, licensed, clinical]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps CVX codes with their corresponding vaccine names and CPT codes. It returns 3 types of vaccine names; `short_name`, `full_name` and `trade_name`.

## Predicted Entities

`short_name`, `full_name`, `trade_name`, `cpt_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/cvx_code_mapper_en_4.2.1_3.0_1665598034618.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .pretrained("cvx_code_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["short_name", "full_name", "trade_name", "cpt_code"])


mapper_pipeline = Pipeline(stages=[
    document_assembler,
    chunk_assembler,
    chunkerMapper
])

data = spark.createDataFrame([['75'], ['20'], ['48'], ['19']]).toDF('text')

res = mapper_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("doc")

val chunk_assembler = new Doc2Chunk()
     .setInputCols(Array("doc"))
     .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("cvx_code_mapper", "en","clinical/models")
     .setInputCols(Array("ner_chunk"))
     .setOutputCol("mappings")
     .setRels(Array("short_name", "full_name", "trade_name", "cpt_code"))

val pipeline = new Pipeline(stages = Array(
     documentAssembler,
     chunk_assembler,
     chunkerMapper))

val data = Seq("75", "20", "48", "19").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------+---------------------+-------------------------------------------------------------+------------+--------+
|cvx_code|short_name           |full_name                                                    |trade_name  |cpt_code|
+--------+---------------------+-------------------------------------------------------------+------------+--------+
|[75]    |[vaccinia (smallpox)]|[vaccinia (smallpox) vaccine]                                |[DRYVAX]    |[90622] |
|[20]    |[DTaP]               |[diphtheria, tetanus toxoids and acellular pertussis vaccine]|[ACEL-IMUNE]|[90700] |
|[48]    |[Hib (PRP-T)]        |[Haemophilus influenzae type b vaccine, PRP-T conjugate]     |[ACTHIB]    |[90648] |
|[19]    |[BCG]                |[Bacillus Calmette-Guerin vaccine]                           |[MYCOBAX]   |[90585] |
+--------+---------------------+-------------------------------------------------------------+------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cvx_code_mapper|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|12.3 KB|
