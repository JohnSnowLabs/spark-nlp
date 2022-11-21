---
layout: model
title: RxNorm Sbd ChunkResolver
author: John Snow Labs
name: chunkresolve_rxnorm_sbd_clinical
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_sbd_clinical_en_3.0.0_3.0_1618603306546.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
rxnorm_resolver = ChunkEntityResolverModel()\
    .pretrained('chunkresolve_rxnorm_sbd_clinical', 'en', "clinical/models")\
    .setEnableLevenshtein(True)\
    .setNeighbours(200).setAlternatives(5).setDistanceWeights([3,11,0,0,0,9])\
    .setInputCols(['token', 'chunk_embeddings'])\
    .setOutputCol('rxnorm_resolution')\
    .setPoolingStrategy("MAX")

pipeline_rxnorm = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, rxnorm_resolver])

model = pipeline_rxnorm.fit(spark.createDataFrame([['']]).toDF("text"))

results = model.transform(data)
```
```scala
...
val rxnorm_resolver = ChunkEntityResolverModel()
    .pretrained('chunkresolve_rxnorm_sbd_clinical', 'en', "clinical/models")
    .setEnableLevenshtein(True)
    .setNeighbours(200).setAlternatives(5).setDistanceWeights(Array(3,11,0,0,0,9))
    .setInputCols('token', 'chunk_embeddings')
    .setOutputCol('rxnorm_resolution')
    .setPoolingStrategy("MAX")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, rxnorm_resolver))

val result = pipeline.fit(Seq.empty[String]).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                          chunk|   entity|                                                                                 target_text(rxnorm)|   code|confidence|
+-----------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
|                                      metformin|TREATMENT|Metformin hydrochloride 500 MG Oral Tablet [Glucamet]:::Metformin hydrochloride 850 MG Oral Table...| 105376|    0.2067|
|                                      glipizide|TREATMENT|Glipizide 5 MG Oral Tablet [Minidiab]:::Glipizide 5 MG Oral Tablet [Glucotrol]:::Glipizide 5 MG O...| 105373|    0.2224|
|                         dapagliflozin for T2DM|TREATMENT|dapagliflozin 5 MG / saxagliptin 5 MG Oral Tablet [Qtern]:::dapagliflozin 10 MG / saxagliptin 5 M...|2169276|    0.2532|
|           atorvastatin and gemfibrozil for HTG|TREATMENT|atorvastatin 20 MG / ezetimibe 10 MG Oral Tablet [Liptruzet]:::atorvastatin 40 MG / ezetimibe 10 ...|1422095|    0.2183|
|                                  dapagliflozin|TREATMENT|dapagliflozin 5 MG Oral Tablet [Farxiga]:::dapagliflozin 10 MG Oral Tablet [Farxiga]:::dapagliflo...|1486981|    0.3523|
|                                    bicarbonate|TREATMENT|Sodium Bicarbonate 0.417 MEQ/ML Oral Solution [Desempacho]:::potassium bicarbonate 25 MEQ Efferve...|1305099|    0.2149|
|insulin drip for euDKA and HTG with a reduction|TREATMENT|insulin aspart, human 30 UNT/ML / insulin degludec 70 UNT/ML Pen Injector [Ryzodeg]:::3 ML insuli...|1994318|    0.2124|
|                                SGLT2 inhibitor|TREATMENT|C1 esterase inhibitor (human) 500 UNT Injection [Cinryze]:::alpha 1-proteinase inhibitor, human 1...| 809871|    0.2044|
|                               insulin glargine|TREATMENT|Insulin Glargine 100 UNT/ML Pen Injector [Lantus]:::Insulin Glargine 300 UNT/ML Pen Injector [Tou...|1359856|    0.2265|
|                      insulin lispro with meals|TREATMENT|Insulin Lispro 100 UNT/ML Cartridge [Humalog]:::Insulin Lispro 200 UNT/ML Pen Injector [Humalog]:...|1652648|    0.2469|
|                                      metformin|TREATMENT|Metformin hydrochloride 500 MG Oral Tablet [Glucamet]:::Metformin hydrochloride 850 MG Oral Table...| 105376|    0.2067|
|                               SGLT2 inhibitors|TREATMENT|alpha 1-proteinase inhibitor, human 1 MG Injection [Prolastin]:::C1 esterase inhibitor (human) 50...|1661220|    0.2167|
+-----------------------------------------------+---------+----------------------------------------------------------------------------------------------------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_rxnorm_sbd_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[rxnorm]|
|Language:|en|
