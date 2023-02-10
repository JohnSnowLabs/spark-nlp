---
layout: model
title: Sentence Entity Resolver for RxNorm (sbluebert_base_uncased_mli embeddings)
author: John Snow Labs
name: sbluebertresolve_rxnorm_augmented_uncased
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

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbluebert_base_uncased_mli` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbluebertresolve_rxnorm_augmented_uncased_en_3.3.4_2.4_1640698320751.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbluebertresolve_rxnorm_augmented_uncased_en_3.3.4_2.4_1640698320751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbluebert_base_uncased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_rxnorm_augmented_uncased", "en", "clinical/models") \
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
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("ner_chunk")
      
val sbert_embedder = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_rxnorm_augmented_uncased", "en", "clinical/models") 
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) 
      .setOutputCol("rxnorm_code")
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(rxnorm_pipelineModel)

val result = light_model.fullAnnotate(Array("Coumadin 5 mg", "aspirin", "avandia 4 mg"))
```
</div>

## Results

```bash
|    |   RxNormCode | Resolution                                 | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-------------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]            | 855333:::432467:::438740:::855... | 0.0000:::1.6841:::1.6841:::3.2... | 0.0000:::0.0062:::0.0062:::0.0... | warfarin sodium 5 MG [Coumadin]:::coumarin 5 MG Oral Tablet:... | Branded Drug Comp:::Clinical D... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet           | 1537020:::1191:::405403:::1001... | 0.0000:::0.0000:::6.0493:::6.4... | 0.0000:::0.0000:::0.0797:::0.0... | aspirin Effervescent Oral Tablet:::aspirin:::YSP Aspirin:::E... | Clinical Drug Form:::Ingredien... |
|  2 |       105029 | gabapentin 300 MG Oral Capsule [Neurontin] | 105029:::1098609:::207088:::20... | 3.1683:::6.0071:::6.2050:::6.2... | 0.0227:::0.0815:::0.0862:::0.0... | gabapentin 300 MG Oral Capsule [Neurontin]:::lamotrigine 300... | Branded Drug:::Branded Drug Co... |
|  3 |       261242 | rosiglitazone 4 MG Oral Tablet [Avandia]   | 261242:::847706:::577784:::212... | 0.0000:::6.8783:::6.9828:::7.4... | 0.0000:::0.1135:::0.1183:::0.1... | rosiglitazone 4 MG Oral Tablet [Avandia]:::glimepiride 4 MG ... | Branded Drug:::Branded Drug Co... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbluebertresolve_rxnorm_augmented_uncased|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|978.4 MB|
|Case sensitive:|false|