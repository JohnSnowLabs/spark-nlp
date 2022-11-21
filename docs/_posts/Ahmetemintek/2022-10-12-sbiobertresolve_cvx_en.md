---
layout: model
title: Sentence Entity Resolver for CVX
author: John Snow Labs
name: sbiobertresolve_cvx
date: 2022-10-12
tags: [entity_resolution, cvx, clinical, en, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps vaccine entities to CVX codes using sbiobert_base_cased_mli Sentence Bert Embeddings. Additionally, this model returns status of the vaccine (Active/Inactive/Pending/Non-US) in all_k_aux_labels column.

## Predicted Entities

`CVX Code`, `Status`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cvx_en_4.2.1_3.0_1665597761894.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
 .setInputCol("text")\
 .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
 .setInputCols(["ner_chunk"])\
 .setOutputCol("sbert_embeddings")

cvx_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cvx", "en", "clinical/models")\
 .setInputCols(["ner_chunk", "sbert_embeddings"])\
 .setOutputCol("cvx_code")\
 .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel( stages = [ documentAssembler, sbert_embedder, cvx_resolver ])

light_model = LightPipeline(pipelineModel)

result = light_model.fullAnnotate(["Sinovac", "Moderna", "BIOTHRAX"])
```
```scala
val documentAssembler = new DocumentAssembler()
 .setInputCol("text")
 .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
 .setInputCols(Array("ner_chunk"))
 .setOutputCol("sbert_embeddings")

val cvx_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cvx", "en", "clinical/models")
 .setInputCols(Array("ner_chunk", "sbert_embeddings"))
 .setOutputCol("cvx_code")
 .setDistanceFunction("EUCLIDEAN")

val cvx_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, cvx_resolver))

val light_model = LightPipeline(cvx_pipelineModel)

val result = light_model.fullAnnotate(Array("Sinovac", "Moderna", "BIOTHRAX"))
```
</div>

## Results

```bash
+----------+--------+-------------------------------------------------------+--------+
 |ner_chunk |cvx_code|resolved_text                                          |Status  |
 +----------+--------+-------------------------------------------------------+--------+
 |Sinovac   |511     |COVID-19 IV Non-US Vaccine (CoronaVac, Sinovac)        |Non-US  |
 |Moderna   |227     |COVID-19, mRNA, LNP-S, PF, pediatric 50 mcg/0.5 mL dose|Inactive|
 |BIOTHRAX  |24      |anthrax                                                |Active  |
 +----------+--------+-------------------------------------------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_cvx|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[bert_embeddings]|
|Output Labels:|[cvx_code]|
|Language:|en|
|Size:|1.6 MB|
|Case sensitive:|false|
