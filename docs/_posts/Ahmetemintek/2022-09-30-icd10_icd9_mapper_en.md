---
layout: model
title: Mapping ICD-10-CM Codes with Their Corresponding ICD-9-CM Codes
author: John Snow Labs
name: icd10_icd9_mapper
date: 2022-09-30
tags: [en, licensed, icd10, icd9, chunk_mapping]
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

This pretrained model maps ICD-10-CM codes to corresponding ICD-9-CM codes

## Predicted Entities

`icd9_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10_icd9_mapper_en_4.1.0_3.0_1664526779493.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10_icd9_mapper_en_4.1.0_3.0_1664526779493.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc", "en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel.pretrained("icd10_icd9_mapper", "en", "clinical/models")\
    .setInputCols(["icd10cm_code"])\
    .setOutputCol("mappings")\
    .setRels(["icd9_code"])


pipeline = Pipeline(stages = [
    documentAssembler,
    sbert_embedder,
    icd_resolver,
    chunkerMapper])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("Diabetes Mellitus")


```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
    .setInputCols(Array("ner_chunk"))
    .setOutputCol("sbert_embeddings")

val icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc", "en", "clinical/models") 
    .setInputCols(Array("ner_chunk", "sbert_embeddings")) 
    .setOutputCol("icd10cm_code")
    .setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel.pretrained("icd10_icd9_mapper", "en", "clinical/models")
    .setInputCols(Array("icd10cm_code"))
    .setOutputCol("mappings")
    .setRels(Array("icd9_code"))


val pipeline = new Pipeline(stages = Array(
    documentAssembler,
    sbert_embedder,
    icd_resolver,
    chunkerMapper
    ))

val data = Seq("Diabetes Mellitus").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)


```
</div>

## Results

```bash
|    | chunk             | icd10cm_code   | icd9_mapping   |
|---:|:------------------|:---------------|:---------------|
|  0 | Diabetes Mellitus | Z833           | V180           |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10_icd9_mapper|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|580.0 KB|
