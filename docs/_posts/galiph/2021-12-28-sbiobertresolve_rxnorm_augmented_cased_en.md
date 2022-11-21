---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_jsl_cased embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_augmented_cased
date: 2021-12-28
tags: [en, clinical, entity_resolution, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_jsl_cased` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_cased_en_3.3.4_2.4_1640687886477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_jsl_cased', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented_cased", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])
light_model = LightPipeline(rxnorm_pipelineModel)

result = light_model.fullAnnotate(["Coumadin 5 mg", "aspirin", "Neurontin 300"])
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("ner_chunk")
      
val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_jsl_cased", "en", "clinical/models")
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented_cased", "en", "clinical/models") 
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) 
      .setOutputCol("rxnorm_code")
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(rxnorm_pipelineModel)

val result = light_model.fullAnnotate(Array("Coumadin 5 mg", "aspirin", "Neurontin 300"))
```
</div>

## Results

```bash
|    |   RxNormCode | Resolution                                 | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-------------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]            | 855333:::645146:::432467:::438... | 7.1909:::8.2961:::8.3727:::8.3... | 0.0887:::0.1170:::0.1176:::0.1... | warfarin sodium 5 MG [Coumadin]:::minoxidil 50 MG/ML Topical... | Branded Drug Comp:::Clinical D... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet           | 1537020:::1191:::437779:::7244... | 0.0000:::0.0000:::8.2570:::8.8... | 0.0000:::0.0000:::0.1147:::0.1... | aspirin Effervescent Oral Tablet:::aspirin:::aspirin / sulfu... | Clinical Drug Form:::Ingredien... |
|  2 |       105029 | gabapentin 300 MG Oral Capsule [Neurontin] | 105029:::2180332:::105852:::19... | 8.7466:::10.7744:::11.1256:::1... | 0.1212:::0.1843:::0.1981:::0.2... | gabapentin 300 MG Oral Capsule [Neurontin]:::darolutamide 30... | Branded Drug:::Branded Drug Co... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_augmented_cased|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|972.4 MB|
|Case sensitive:|false|
