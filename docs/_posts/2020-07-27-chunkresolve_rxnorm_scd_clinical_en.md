---
layout: model
title: RxNorm Scd ChunkResolver
author: John Snow Labs
name: chunkresolve_rxnorm_scd_clinical
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_scd_clinical_en_2.5.1_2.4_1595813884363.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

{:.h2_title}
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
|----------------|----------------------------------|
| Name:           | chunkresolve_rxnorm_scd_clinical |
| Type:    | ChunkEntityResolverModel         |
| Compatibility:  | Spark NLP 2.5.1+                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [token, chunk_embeddings]          |
|Output labels:        | [entity]                           |
| Language:       | en                               |
| Case sensitive: | True                             |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on December 2019 RxNorm Clinical Drugs (TTY=SCD) ontology graph with `embeddings_clinical`
https://www.nlm.nih.gov/pubs/techbull/nd19/brief/nd19_rxnorm_december_2019_release.html
