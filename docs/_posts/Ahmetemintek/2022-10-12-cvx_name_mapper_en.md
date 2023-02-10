---
layout: model
title: Mapping Vaccine Products with Their Corresponding CVX Codes, Vaccine Names and CPT Codes
author: John Snow Labs
name: cvx_name_mapper
date: 2022-10-12
tags: [cvx, chunk_mapping, cpt, en, clinical, licensed]
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

This pretrained model maps vaccine products with their corresponding CVX codes, vaccine names and CPT codes. It returns 3 types of vaccine names; `short_name`, `full_name` and `trade_name`.

## Predicted Entities

`cvx_code`, `short_name`, `full_name`, `trade_name`, `cpt_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/cvx_name_mapper_en_4.2.1_3.0_1665599269592.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/cvx_name_mapper_en_4.2.1_3.0_1665599269592.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .pretrained("cvx_name_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["cvx_code", "short_name", "full_name", "trade_name", "cpt_code"])


mapper_pipeline = Pipeline(stages=[
    document_assembler,
    chunk_assembler,
    chunkerMapper
])

data = spark.createDataFrame([['DTaP'], ['MYCOBAX'], ['cholera, live attenuated']]).toDF('text')

res = mapper_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("doc")

val chunk_assembler = new Doc2Chunk()
  .setInputCols(Array("doc"))
  .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("cvx_name_mapper", "en","clinical/models")
  .setInputCols(Array("ner_chunk"))
  .setOutputCol("mappings")
  .setRels(Array("cvx_code", "short_name", "full_name", "trade_name", "cpt_code"))

val pipeline = new Pipeline(stages = Array(
  documentAssembler,
  chunk_assembler,
  chunkerMapper))

val data = Seq("DTaP", "MYCOBAX", "cholera, live attenuated").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
|chunk                     |cvx_code|short_name                |full_name                                                    |trade_name  |cpt_code|
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
|[DTaP]                    |[20]    |[DTaP]                    |[diphtheria, tetanus toxoids and acellular pertussis vaccine]|[ACEL-IMUNE]|[90700] |
|[MYCOBAX]                 |[19]    |[BCG]                     |[Bacillus Calmette-Guerin vaccine]                           |[MYCOBAX]   |[90585] |
|[cholera, live attenuated]|[174]   |[cholera, live attenuated]|[cholera, live attenuated]                                   |[VAXCHORA]  |[90625] |
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cvx_name_mapper|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|25.1 KB|
