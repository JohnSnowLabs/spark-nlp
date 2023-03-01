---
layout: model
title: Mapping National Drug Codes (NDC) Codes with Corresponding Drug Brand Names
author: John Snow Labs
name: ndc_drug_brandname_mapper
date: 2023-02-22
tags: [chunk_mapping, ndc, drug_brand_name, clinical, en, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps National Drug Codes (NDC) codes with their corresponding drug brand names.

## Predicted Entities

`drug_brand_name`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ndc_drug_brandname_mapper_en_4.3.0_3.0_1677102197072.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ndc_drug_brandname_mapper_en_4.3.0_3.0_1677102197072.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

mapper = DocMapperModel.pretrained("ndc_drug_brandname_mapper", "en", "clinical/models")\
    .setInputCols("document")\
    .setOutputCol("mappings")\
    .setRels(["drug_brand_name"])\

pipeline = Pipeline(
    stages = [
        documentAssembler,
        mapper
        ])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

lp = LightPipeline(model)

result = lp.fullAnnotate(["0009-4992", "57894-150"])
```
```scala
val documentAssembler = new DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

val mapper = DocMapperModel.pretrained("ndc_drug_brandname_mapper", "en", "clinical/models")\
    .setInputCols("document")\
    .setOutputCol("mappings")\
    .setRels(Array("drug_brand_name")\

val pipeline = new Pipeline(stages = Array(
        documentAssembler,
        mapper
))

val data = Seq(Array("0009-4992", "57894-150")).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | ndc_code   | drug_brand_name   |
|---:|:-----------|:------------------|
|  0 | 0009-4992  | ZYVOX             |
|  1 | 57894-150  | ZYTIGA            |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ndc_drug_brandname_mapper|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[brandname]|
|Language:|en|
|Size:|917.7 KB|