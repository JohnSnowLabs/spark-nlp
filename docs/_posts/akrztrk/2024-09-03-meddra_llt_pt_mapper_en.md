---
layout: model
title: Mapping MedDRA-LLT (Lowest Level Term) Codes With Their Corresponding MedDRA-PT (Preferred Term) Codes
author: John Snow Labs
name: meddra_llt_pt_mapper
date: 2024-09-03
tags: [licensed, en, meddra, llt, pt, mapping, open_source]
task: Chunk Mapping
language: en
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps MedDRA LLT (Lowest Level Term) to corresponding MedDRA PT (Preferred Term) codes.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/meddra_llt_pt_mapper_en_5.4.1_3.0_1725381134975.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/meddra_llt_pt_mapper_en_5.4.1_3.0_1725381134975.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('doc')

chunk_assembler = Doc2Chunk()\
      .setInputCols(['doc'])\
      .setOutputCol('chunk')
 
mapperModel = ChunkMapperModel.load('meddra_llt_pt_mapper')\
    .setInputCols(["chunk"])\
    .setOutputCol("mappings")\
    .setRels(["pt_code"])


pipeline = Pipeline(stages=[
    document_assembler,
    chunk_assembler,
    mapperModel
])

data = spark.createDataFrame([["10002442"], ["10000007"], ["10003696"]]).toDF("text")

mapper_model = pipeline.fit(data)
result = mapper_model.transform(data)
```
```scala
val document_assembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

val chunk_assembler = Doc2Chunk()
      .setInputCols(Array("doc"))
      .setOutputCol("chunk")
 
val mapperModel = ChunkMapperModel.load("meddra_llt_pt_mapper")
    .setInputCols(Array("chunk"))
    .setOutputCol("mappings")
    .setRels(["pt_code"])


val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    chunk_assembler,
    mapperModel))

val data = Seq("10002442", "10000007", "10003696").toDF("text")

val mapper_model = pipeline.fit(data)
val result = mapper_model.transform(data)
```
</div>

## Results

```bash
+--------+----------------------------------------+
|llt_code|pt_code                                 |
+--------+----------------------------------------+
|10002442|10002442:Angiogram pulmonary normal     |
|10000007|10000007:17 ketosteroids urine decreased|
|10003696|10001324:Adrenal atrophy                |
+--------+----------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|meddra_llt_pt_mapper|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.9 MB|

## References

This model is trained with the September 2024 (v27.1) release of MedDRA dataset.

**To utilize this model, possession of a valid MedDRA license is requisite. If you possess one and wish to use this model, kindly contact us at support@johnsnowlabs.com.**