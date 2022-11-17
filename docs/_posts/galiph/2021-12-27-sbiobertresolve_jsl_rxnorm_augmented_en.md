---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_jsl_rxnorm_cased embeddings)
author: John Snow Labs
name: sbiobertresolve_jsl_rxnorm_augmented
date: 2021-12-27
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

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_jsl_rxnorm_cased` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_jsl_rxnorm_augmented_en_3.3.4_2.4_1640637079907.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_jsl_rxnorm_cased', 'en','clinical/models')\
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
light_model = LightPipeline(rxnorm_pipelineModel)

result = light_model.fullAnnotate(["Coumadin 5 mg", "aspirin", "Neurontin 300", "avandia 4 mg"])
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("ner_chunk")
      
val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_jsl_rxnorm_cased", "en", "clinical/models")
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_jsl_rxnorm_augmented", "en", "clinical/models") 
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
|    |   RxNormCode | Resolution                                                         | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-------------------------------------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]                                    | 855333:::432467:::438740:::103... | 0.0000:::5.0617:::5.0617:::5.9... | 0.0000:::0.0388:::0.0388:::0.0... | warfarin sodium 5 MG [Coumadin]:::coumarin 5 MG Oral Tablet:... | Branded Drug Comp:::Clinical D... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet                                   | 1537020:::1191:::405403:::2187... | 0.0000:::0.0000:::9.0615:::9.4... | 0.0000:::0.0000:::0.1268:::0.1... | aspirin Effervescent Oral Tablet:::aspirin:::YSP Aspirin:::N... | Clinical Drug Form:::Ingredien... |
|  2 |       261242 | rosiglitazone 4 MG Oral Tablet [Avandia]                           | 261242:::208364:::1792373:::57... | 0.0000:::8.0227:::8.1631:::8.2... | 0.0000:::0.0982:::0.1001:::0.1... | rosiglitazone 4 MG Oral Tablet [Avandia]:::triamcinolone 4 M... | Branded Drug:::Branded Drug:::... |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_jsl_rxnorm_augmented|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|970.8 MB|
|Case sensitive:|false|
