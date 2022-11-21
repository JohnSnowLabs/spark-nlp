---
layout: model
title: Mapping ICD10-CM Codes with Their Corresponding SNOMED Codes
author: John Snow Labs
name: icd10cm_snomed_mapper
date: 2022-06-26
tags: [icd10cm, snomed, clinical, en, chunk_mapper, licensed]
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

This pretrained model maps ICD10-CM codes to corresponding SNOMED codes under the Unified Medical Language System (UMLS).

## Predicted Entities

`snomed_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_snomed_mapper_en_3.5.3_3.0_1656230731120.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

chunkerMapper = ChunkMapperModel.pretrained("icd10cm_snomed_mapper", "en", "clinical/models")\
.setInputCols(["icd10cm_code"])\
.setOutputCol("mappings")\
.setRels(["snomed_code"])


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

val chunkerMapper = ChunkMapperModel.pretrained("icd10cm_snomed_mapper", "en","clinical/models")
.setInputCols(Array("icd10cm_code"))
.setOutputCol("mappings")
.setRels(Array("snomed_code"))


val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
icd_resolver,
chunkerMapper))


val data = Seq("Diabetes Mellitus").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.icd10cm_to_snomed").predict("""Diabetes Mellitus""")
```

</div>

## Results

```bash
|    | ner_chunk         | icd10cm_code   |   snomed_mappings |
|---:|:------------------|:---------------|------------------:|
|  0 | Diabetes Mellitus | Z833           |         160402005 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_snomed_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[icd10_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.1 MB|

