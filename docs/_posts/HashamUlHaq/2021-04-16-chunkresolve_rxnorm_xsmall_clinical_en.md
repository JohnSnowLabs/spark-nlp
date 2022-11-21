---
layout: model
title: RxNorm Xsmall ChunkResolver
author: John Snow Labs
name: chunkresolve_rxnorm_xsmall_clinical
date: 2021-04-16
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

RxNorm Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_xsmall_clinical_en_3.0.0_3.0_1618603394135.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
rxnorm_resolver = ChunkEntityResolverModel()\
    .pretrained('chunkresolve_rxnorm_xsmall_clinical', 'en', "clinical/models")\
    .setEnableLevenshtein(True)\
    .setNeighbours(200).setAlternatives(5).setDistanceWeights([3,11,0,0,0,9])\
    .setInputCols(['token', 'chunk_embeddings'])\
    .setOutputCol('rxnorm_resolution')\
    .setPoolingStrategy("MAX")
pipeline_rxnorm = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, rxnorm_resolver])

model = pipeline_rxnorm.fit(spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation."""]]).toDF("text"))

results = model.transform(data)
```
```scala
...
val rxnorm_resolver = ChunkEntityResolverModel()
    .pretrained('chunkresolve_rxnorm_xsmall_clinical', 'en', "clinical/models")
    .setEnableLevenshtein(True)
    .setNeighbours(200).setAlternatives(5).setDistanceWeights(Array(3,11,0,0,0,9))
    .setInputCols('token', 'chunk_embeddings')
    .setOutputCol('rxnorm_resolution')
    .setPoolingStrategy("MAX")
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, rxnorm_resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                                          chunk|   entity|                                                                                         target_text|   code|confidence|
+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                                      metformin|TREATMENT|Glipizide Metformin hydrochloride:::Glyburide Metformin hydrochloride:::Glipizide Metformin hydro...| 861731|    0.2000|
|                                                      glipizide|TREATMENT|                   Glipizide:::Glipizide:::Glipizide:::Glipizide:::Glipizide Metformin hydrochloride| 310488|    0.2499|
|dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG|TREATMENT|dapagliflozin saxagliptin:::dapagliflozin saxagliptin:::dapagliflozin saxagliptin:::dapagliflozin...|1925504|    0.2080|
|                                                  dapagliflozin|TREATMENT|           dapagliflozin:::dapagliflozin:::dapagliflozin:::dapagliflozin:::dapagliflozin saxagliptin|1488574|    0.2492|
+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_rxnorm_xsmall_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[rxnorm]|
|Language:|en|
