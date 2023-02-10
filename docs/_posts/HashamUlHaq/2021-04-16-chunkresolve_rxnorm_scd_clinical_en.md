---
layout: model
title: RxNorm Scd ChunkResolver
author: John Snow Labs
name: chunkresolve_rxnorm_scd_clinical
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_scd_clinical_en_3.0.0_3.0_1618603397185.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_scd_clinical_en_3.0.0_3.0_1618603397185.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

rxnormResolver = ChunkEntityResolverModel()\
    .pretrained('chunkresolve_rxnorm_scd_clinical', 'en', "clinical/models")\
    .setEnableLevenshtein(True)\
    .setNeighbours(200).setAlternatives(5).setDistanceWeights([3,3,2,0,0,7])\
    .setInputCols(['token', 'chunk_embs_drug'])\
    .setOutputCol('rxnorm_resolution')\

pipeline_rxnorm = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, jslNer, drugNer, jslConverter, drugConverter, jslChunkEmbeddings, drugChunkEmbeddings, rxnormResolver])

model = pipeline_rxnorm.fit(spark.createDataFrame([['']]).toDF("text"))

results = model.transform(data)
```
```scala
...

val rxnormResolver = ChunkEntityResolverModel()
    .pretrained('chunkresolve_rxnorm_scd_clinical', 'en', "clinical/models")
    .setEnableLevenshtein(True)
    .setNeighbours(200).setAlternatives(5).setDistanceWeights(Array(3,3,2,0,0,7))
    .setInputCols('token', 'chunk_embs_drug')
    .setOutputCol('rxnorm_resolution')

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, jslNer, drugNer, jslConverter, drugConverter, jslChunkEmbeddings, drugChunkEmbeddings, rxnormResolver))

val result = pipeline.fit(Seq.empty[String]).transform(data)
```
</div>

## Results

```bash
| coords       | chunk       | entity    | rxnorm_opts                                                                             |
|--------------|-------------|-----------|-----------------------------------------------------------------------------------------|
| 3::278::287  | creatinine  | DrugChem  | [(849628, Creatinine 800 MG Oral Capsule), (252180, Urea 10 MG/ML Topical Lotion), ...] |
| 7::83::93    | cholesterol | DrugChem  | [(2104173, beta Sitosterol 35 MG Oral Tablet), (832876, phytosterol esters 500 MG O...] |
| 10::397::406 | creatinine  | DrugChem  | [(849628, Creatinine 800 MG Oral Capsule), (252180, Urea 10 MG/ML Topical Lotion), ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_rxnorm_scd_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[rxnorm]|
|Language:|en|
