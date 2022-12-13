---
layout: model
title: Mapping SNOMED Codes with Their Corresponding ICD10-CM Codes
author: John Snow Labs
name: snomed_icd10cm_mapper
date: 2022-06-26
tags: [clinical, licensed, icd10cm, chunk_mapper, en, snomed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps SNOMED codes to corresponding ICD10-CM codes.

## Predicted Entities

`icd10cm_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_icd10cm_mapper_en_3.5.3_3.0_1656266755041.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/snomed_icd10cm_mapper_en_3.5.3_3.0_1656266755041.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_uncased", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel.pretrained("snomed_icd10cm_mapper", "en", "clinical/models")\
.setInputCols(["snomed_code"])\
.setOutputCol("icd10cm_mappings")\
.setRels(["icd10cm_code"])

pipeline = Pipeline(
stages = [
documentAssembler,
sbert_embedder,
snomed_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("Radiating chest pain")
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_uncased", "en", "clinical/models")
.setInputCols("ner_chunk")
.setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("snomed_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel.pretrained("snomed_icd10cm_mapper", "en", "clinical/models")
.setInputCols("snomed_code")
.setOutputCol("icd10cm_mappings")
.setRels(Array("icd10cm_code"))

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
snomed_resolver,
chunkerMapper))

val data = Seq("Radiating chest pain").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.snomed_to_icd10cm").predict("""Radiating chest pain""")
```

</div>

## Results

```bash
|    | ner_chunk            |   snomed_code | icd10cm_mappings   |
|---:|:---------------------|--------------:|:-------------------|
|  0 | Radiating chest pain |      10000006 | R07.9              |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_icd10cm_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[snomed_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.5 MB|

## References

This pretrained model maps SNOMED codes to corresponding  ICD10-CM codes under the Unified Medical Language System (UMLS).