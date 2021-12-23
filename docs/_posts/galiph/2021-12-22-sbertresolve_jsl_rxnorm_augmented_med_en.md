---
layout: model
title: Sentence Entity Resolver for RxNorm (jsl_sbert_medium_rxnorm embeddings)
author: John Snow Labs
name: sbertresolve_jsl_rxnorm_augmented_med
date: 2021-12-22
tags: [rxnorm, licensed, en, clinical, entity_resolution, open_source]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `jsl_sbert_medium_rxnorm` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbertresolve_jsl_rxnorm_augmented_med_en_3.3.4_2.4_1640152257777.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('jsl_sbert_medium_rxnorm', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_jsl_rxnorm_augmented_med", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])

light_model = LightPipeline(pipelineModel)
result = light_model.fullAnnotate(['folic acid', 'levothyroxine', 'aspirin', 'magnesium citrate',])

```
```scala
val documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained('jsl_sbert_medium_rxnorm', 'en','clinical/models')\
      .setInputCols("ner_chunk")\
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_jsl_rxnorm_augmented_med", "en", "clinical/models") \
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(pipelineModel)
val result = light_model.fullAnnotate(['folic acid', 'levothyroxine', 'aspirin', 'magnesium citrate'])

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk             |   RxNormCode | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                 | all_k_aux_labels                  |
|---:|:------------------|-------------:|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|
|  0 | folic acid        |       619037 | 619037:::4511:::62356:::144086... | 0.0000:::0.0000:::9.6048:::9.9... | 0.0000:::0.0000:::0.1439:::0.1... | folic acid :::folic acid:::fol... | Clinical Drug Form:::Ingredien... |
|  1 | levothyroxine     |        10582 | 10582:::1001569:::3292:::37177... | 0.0000:::0.0000:::7.9957:::7.9... | 0.0000:::0.0000:::0.1006:::0.1... | levothyroxine:::levothyroxine ... | Ingredient:::Clinical Drug For... |
|  2 | aspirin           |      1537020 | 1537020:::1191:::405403:::2187... | 0.0000:::0.0000:::9.0615:::9.4... | 0.0000:::0.0000:::0.1268:::0.1... | aspirin :::aspirin:::ysp aspir... | Clinical Drug Form:::Ingredien... |
|  3 | magnesium citrate |       204730 | 204730:::52356:::314718:::1314... | 0.0000:::0.0000:::5.0972:::5.7... | 0.0000:::0.0000:::0.0398:::0.0... | magnesium citrate :::magnesium... | Clinical Drug Form:::Ingredien... |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_jsl_rxnorm_augmented_med|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|650.9 MB|
|Case sensitive:|false|
