---
layout: model
title: Mapping ICDO Codes with Their Corresponding SNOMED Codes
author: John Snow Labs
name: icdo_snomed_mapper
date: 2022-06-26
tags: [icdo, snomed, chunk_mapper, clinical, licensed, en]
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

This pretrained model maps ICDO codes to corresponding SNOMED codes.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icdo_snomed_mapper_en_3.5.3_3.0_1656274513770.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.setOutputCol("sbert_embeddings")

icdo_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_icdo_augmented", "en", "clinical/models")\
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("icdo_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel\
.pretrained("icdo_snomed_mapper", "en", "clinical/models")\
.setInputCols(["icdo_code"])\
.setOutputCol("snomed_mappings")\
.setRels(["snomed_code"])


pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
icdo_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("Hepatocellular Carcinoma")
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val icdo_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icdo_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("icdo_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel
.pretrained("icdo_snomed_mapper", "en", "clinical/models")
.setInputCols(Array("icdo_code"))
.setOutputCol("snomed_mappings")
.setRels(Array("snomed_code"))

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
icdo_resolver,
chunkerMapper
))

val data = Seq("Hepatocellular Carcinoma").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.icdo_to_snomed").predict("""Hepatocellular Carcinoma""")
```

</div>

## Results

```bash
|    | ner_chunk                | icdo_code   |   snomed_mappings |
|---:|:-------------------------|:------------|------------------:|
|  0 | Hepatocellular Carcinoma | 8170/3      |          25370001 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icdo_snomed_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[icdo_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|127.9 KB|

## References

This pretrained model maps ICDO codes to corresponding SNOMED codes under the Unified Medical Language System (UMLS).
