---
layout: model
title: Mapping MESH Codes with Their Corresponding UMLS Codes
author: John Snow Labs
name: mesh_umls_mapper
date: 2022-06-26
tags: [mesh, umls, chunk_mapper, clinical, licensed, en]
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

This pretrained model maps MESH codes to corresponding UMLS codes under the Unified Medical Language System (UMLS).

## Predicted Entities

`umls_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/mesh_umls_mapper_en_3.5.3_3.0_1656281333787.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")\
.setCaseSensitive(False)

mesh_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_mesh", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("mesh_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel\
.pretrained("mesh_umls_mapper", "en", "clinical/models")\
.setInputCols(["mesh_code"])\
.setOutputCol("umls_mappings")\
.setRels(["umls_code"])


pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
mesh_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("N-acetyl-L-arginine")
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols("ner_chunk")
.setOutputCol("sbert_embeddings")

val mesh_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_mesh", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("mesh_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel
.pretrained("mesh_umls_mapper", "en", "clinical/models")
.setInputCols("mesh_code")
.setOutputCol("umls_mappings")
.setRels(Array("umls_code"))

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
mesh_resolver,
chunkerMapper
))

val data = Seq("N-acetyl-L-arginine").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.mesh_to_umls").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | ner_chunk           | mesh_code   | umls_mappings   |
|---:|:--------------------|:------------|:----------------|
|  0 | N-acetyl-L-arginine | C000015     | C0067655        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mesh_umls_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[mesh_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|3.8 MB|