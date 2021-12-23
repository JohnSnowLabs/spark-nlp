---
layout: model
title: Sentence Entity Resolver for RxNorm (jsl_sbiobert_rxnorm embeddings)
author: John Snow Labs
name: sbiobertresolve_jsl_rxnorm_augmented
date: 2021-12-23
tags: [en, clinical, entity_resolution, licensed, open_source]
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

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `jsl_sbiobert_rxnorm` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbiobertresolve_jsl_rxnorm_augmented_en_3.3.4_2.4_1640255258310.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")
sbert_embedder = BertSentenceEmbeddings.pretrained('jsl_sbiobert_rxnorm', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_jsl_rxnorm_augmented", "en", "clinical/models") \
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

val sbert_embedder = BertSentenceEmbeddings.pretrained('jsl_sbiobert_rxnorm', 'en','clinical/models')\
      .setInputCols("ner_chunk")\
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_jsl_rxnorm_augmented", "en", "clinical/models") \
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(pipelineModel)
val result = light_model.fullAnnotate(['folic acid', 'levothyroxine', 'aspirin', 'magnesium citrate'])
```
</div>

## Results

```bash
|    | chunk             |   RxNormCode | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                 | all_k_aux_labels                  |
|---:|:------------------|-------------:|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|
|  0 | folic acid        |       219962 | 219962:::1424174:::1745384:::4... | 15.4317:::15.7788:::15.7788:::... | 0.3777:::0.3696:::0.3696:::0.3... | Slow Fe with Folic Acid:::radi... | Brand Name:::Ingredient:::Clin... |
|  1 | levothyroxine     |      1050068 | 1050068:::96459:::214980:::239... | 14.0105:::14.0118:::14.4574:::... | 0.3236:::0.3078:::0.3152:::0.3... | delos brand of benzoyl peroxid... | Brand Name:::Brand Name:::Bran... |
|  2 | aspirin           |      1537020 | 1537020:::1191:::405403:::2200... | 12.4577:::12.4577:::13.7608:::... | 0.2375:::0.2375:::0.3044:::0.3... | aspirin :::aspirin:::ysp aspir... | Clinical Drug Form:::Ingredien... |
|  3 | magnesium citrate |      1007014 | 1007014:::2001679:::441029:::4... | 15.4448:::15.5450:::16.0236:::... | 0.3710:::0.3616:::0.3822:::0.3... | citrate of magnesium 1.745 g/m... | Multiple Ingredients:::Precise... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_jsl_rxnorm_augmented|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|970.9 MB|
|Case sensitive:|false|