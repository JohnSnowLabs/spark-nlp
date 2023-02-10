---
layout: model
title: Mapping ICD10CM Codes with Their Corresponding UMLS Codes
author: John Snow Labs
name: icd10cm_umls_mapper
date: 2022-06-26
tags: [icd10cm, umls, chunk_mapper, clinical, licensed, en]
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

This pretrained model maps ICD10CM codes to corresponding UMLS codes under the Unified Medical Language System (UMLS).

## Predicted Entities

`umls_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_umls_mapper_en_3.5.3_3.0_1656278690210.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10cm_umls_mapper_en_3.5.3_3.0_1656278690210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

icd10cm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_icd10cm","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("icd10cm_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel\
.pretrained("icd10cm_umls_mapper", "en", "clinical/models")\
.setInputCols(["icd10cm_code"])\
.setOutputCol("umls_mappings")\
.setRels(["umls_code"])


pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
icd10cm_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("Neonatal skin infection")
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val icd10cm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel
.pretrained("icd10cm_umls_mapper", "en", "clinical/models")
.setInputCols(Array("rxnorm_code"))
.setOutputCol("umls_mappings")
.setRels(Array("umls_code"))

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
icd10cm_resolver,
chunkerMapper
))

val data = Seq("Neonatal skin infection").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.icd10cm_to_umls").predict("""Neonatal skin infection""")
```

</div>

## Results

```bash
|    | ner_chunk               | icd10cm_code   | umls_mappings   |
|---:|:------------------------|:---------------|:----------------|
|  0 | Neonatal skin infection | P394           | C0456111        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_umls_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[icd10cm_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|942.9 KB|