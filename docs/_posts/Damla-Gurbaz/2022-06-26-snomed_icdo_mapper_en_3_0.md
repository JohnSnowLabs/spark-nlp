---
layout: model
title: Mapping SNOMED Codes with Their Corresponding ICDO Codes
author: John Snow Labs
name: snomed_icdo_mapper
date: 2022-06-26
tags: [snomed, icdo, chunk_mapper, clinical, licensed, en]
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

This pretrained model maps SNOMED codes to corresponding ICDO codes under the Unified Medical Language System (UMLS).

## Predicted Entities

`icdo_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_icdo_mapper_en_3.5.3_3.0_1656279162444.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

snomed_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel\
.pretrained("snomed_icdo_mapper", "en", "clinical/models")\
.setInputCols(["snomed_code"])\
.setOutputCol("icdo_mappings")\
.setRels(["icdo_code"])


pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
snomed_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline= LightPipeline(model)

result = light_pipeline.fullAnnotate("Structure of tendon of gluteus minimus")
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols("ner_chunk")
.setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("snomed_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel
.pretrained("snomed_icdo_mapper", "en", "clinical/models")
.setInputCols("snomed_code")
.setOutputCol("icdo_mappings")
.setRels(Array("icdo_code"))

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
snomed_resolver,
chunkerMapper
))

val data = Seq("Structure of tendon of gluteus minimus").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.snomed_to_icdo").predict("""Structure of tendon of gluteus minimus""")
```

</div>

## Results

```bash
|    | ner_chunk                              | snomed_code |   icdo_mappings |
|---:|:---------------------------------------|:------------|----------------:|
|  0 | Structure of tendon of gluteus minimus | 128501000   |           C49.5 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_icdo_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[snomed_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|203.5 KB|
