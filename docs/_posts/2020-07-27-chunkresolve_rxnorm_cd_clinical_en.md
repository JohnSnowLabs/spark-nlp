---
layout: model
title: RxNorm Cd ChunkResolver
author: John Snow Labs
name: chunkresolve_rxnorm_cd_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-07-27
task: Entity Resolution
edition: Healthcare NLP 2.5.1
spark_version: 2.4
tags: [clinical,licensed,entity_resolution,en]
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities
RxNorm Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_cd_clinical_en_2.5.1_2.4_1595813950836.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_cd_clinical_en_2.5.1_2.4_1595813950836.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
...
rxnorm_resolver = ChunkEntityResolverModel()\
    .pretrained('chunkresolve_rxnorm_cd_clinical', 'en', "clinical/models")\
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
    .pretrained('chunkresolve_rxnorm_cd_clinical', 'en', "clinical/models")
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

{:.h2_title}
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
|----------------|---------------------------------|
| Name:           | chunkresolve_rxnorm_cd_clinical |
| Type:    | ChunkEntityResolverModel        |
| Compatibility:  | Spark NLP 2.5.1+                           |
| License:        | Licensed                        |
|Edition:|Official|                      |
|Input labels:         | [token, chunk_embeddings]         |
|Output labels:        | [entity]                          |
| Language:       | en                              |
| Case sensitive: | True                            |
| Dependencies:  | embeddings_clinical             |

{:.h2_title}
## Data Source
Trained on December 2019 RxNorm Clinical Drugs (TTY=CD) ontology graph with `embeddings_clinical`
https://www.nlm.nih.gov/pubs/techbull/nd19/brief/nd19_rxnorm_december_2019_release.html
