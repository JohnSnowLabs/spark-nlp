---
layout: model
title: ICD10PCS Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10pcs_clinical
date: 2021-04-02
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities

ICD10-PCS Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_3.0.0_3.0_1617355415038.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

pipeline_icd10pcs = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, ner, chunk_embeddings, model])

data = ["""He has a starvation ketosis but nothing found for significant for dry oral mucosa"""]

pipeline_model = pipeline_icd10pcs.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline = LightPipeline(pipeline_model)

result = light_pipeline.annotate(data)
```
```scala
...
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, ner, chunk_embeddings, model))

val data = Seq("He has a starvation ketosis but nothing found for significant for dry oral mucosa").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|   | chunks               | begin | end | code    | resolutions                                      |
|---|----------------------|-------|-----|---------|--------------------------------------------------|
| 0 | a starvation ketosis | 7     | 26  | 6A3Z1ZZ | Hyperthermia, Multiple:::Narcosynthesis:::Hype...|
| 1 | dry oral mucosa      | 66    | 80  | 8E0ZXY4 | Yoga Therapy:::Release Cecum, Open Approach:::...|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_icd10pcs_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10pcs]|
|Language:|en|
