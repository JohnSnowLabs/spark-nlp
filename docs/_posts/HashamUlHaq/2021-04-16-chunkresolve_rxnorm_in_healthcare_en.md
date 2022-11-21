---
layout: model
title: Chunk Entity Resolver RxNorm-scdc
author: John Snow Labs
name: chunkresolve_rxnorm_in_healthcare
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

This model maps extracted medical entities to RxNorm codes using chunk embeddings (augmented with synonyms, four times richer than previous resolver).

## Predicted Entities

RxNorm codes

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_RXNORM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_in_healthcare_en_3.0.0_3.0_1618605195699.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
resolver = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_in_healthcare","en","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, resolver])

data = spark.createDataFrame([["A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation."]]).toDF("text")
model = pipeline.fit(data)
results = model.transform(data)
...

```
```scala

...
val resolver = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_in_healthcare","en","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(document_assembler, sbert_embedder, resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation.").toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash

+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                                          chunk|   entity|                                                                                         target_text|   code|confidence|
+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                                      metformin|TREATMENT|metFORMIN compounding powder:::Metformin Hydrochloride Powder:::metFORMIN 500 mg oral tablet:::me...| 601021|    0.2364|
|                                                      glipizide|TREATMENT|Glipizide Powder:::Glipizide Crystal:::Glipizide Tablets:::glipiZIDE 5 mg oral tablet:::glipiZIDE...| 241604|    0.3647|
|dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG|TREATMENT|Ezetimibe and Atorvastatin Tablets:::Amlodipine and Atorvastatin Tablets:::Atorvastatin Calcium T...|1422084|    0.3407|
|                                                  dapagliflozin|TREATMENT|Dapagliflozin Tablets:::dapagliflozin 5 mg oral tablet:::dapagliflozin 10 mg oral tablet:::Dapagl...|1488568|    0.7070|
+---------------------------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_rxnorm_in_healthcare|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
