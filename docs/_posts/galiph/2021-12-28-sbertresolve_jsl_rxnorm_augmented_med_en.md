---
layout: model
title: Sentence Entity Resolver for RxNorm (sbert_jsl_medium_rxnorm_uncased embeddings)
author: John Snow Labs
name: sbertresolve_jsl_rxnorm_augmented_med
date: 2021-12-28
tags: [clinical, entity_resolution, en, licensed]
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

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbert_jsl_medium_rxnorm_uncased` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_jsl_rxnorm_augmented_med_en_3.3.4_2.4_1640686630389.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_rxnorm_uncased', 'en','clinical/models')\
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
light_model = LightPipeline(rxnorm_pipelineModel)

result = light_model.fullAnnotate(["Coumadin 5 mg", "aspirin", "Neurontin 300", "avandia 4 mg"])
```
```scala
val documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")
      
val sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_rxnorm_uncased", "en", "clinical/models")\
      .setInputCols("ner_chunk")\
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_jsl_rxnorm_augmented_med", "en", "clinical/models") \
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(rxnorm_pipelineModel)

val result = light_model.fullAnnotate(Array("Coumadin 5 mg", "aspirin", "Neurontin 300", "avandia 4 mg"))
```
</div>

## Results

```bash
|    |   RxNormCode | Resolution                                 | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-------------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]            | 855333:::855334:::1110792:::11... | 0.0000:::6.0548:::6.1667:::6.1... | 0.0000:::0.0515:::0.0536:::0.0... | warfarin sodium 5 MG [Coumadin]:::warfarin sodium 5 MG Oral ... | Branded Drug Comp:::Branded Dr... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet           | 1537020:::1191:::202547:::1001... | 0.0000:::0.0000:::8.8123:::9.3... | 0.0000:::0.0000:::0.1145:::0.1... | aspirin Effervescent Oral Tablet:::aspirin:::Empirin:::Ecpir... | Clinical Drug Form:::Ingredien... |
|  2 |       105029 | gabapentin 300 MG Oral Capsule [Neurontin] | 105029:::1718929:::1718930:::3... | 5.5969:::8.7502:::8.7502:::8.7... | 0.0452:::0.1092:::0.1092:::0.1... | gabapentin 300 MG Oral Capsule [Neurontin]:::olanzapine 300 ... | Branded Drug:::Clinical Drug C... |
|  3 |       261242 | rosiglitazone 4 MG Oral Tablet [Avandia]   | 261242:::2123140:::1792373:::8... | 0.0000:::7.1217:::7.7113:::8.4... | 0.0000:::0.0728:::0.0843:::0.1... | rosiglitazone 4 MG Oral Tablet [Avandia]:::erdafitinib 4 MG ... | Branded Drug:::Branded Drug Co... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_jsl_rxnorm_augmented_med|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|650.7 MB|
|Case sensitive:|false|
