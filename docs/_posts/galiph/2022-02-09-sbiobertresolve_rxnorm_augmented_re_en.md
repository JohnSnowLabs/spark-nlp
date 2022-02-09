---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli - EntityChunkEmbeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_augmented_re
date: 2022-02-09
tags: [rxnorm, licensed, en, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.4.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes without specifying the relations between the entities (relations are calculated on the fly inside the annotator) using sbiobert_base_cased_mli Sentence Bert Embeddings (EntityChunkEmbeddings). Embeddings used in this model are calculated with following weights : `{"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2}` . EntityChunkEmbeddings with those weights are required in the pipeline to get best result.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_re_en_3.4.0_2.4_1644395696788.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

sentence_detector = SentenceDetector() \
    .setInputCols("documents") \
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols("sentences") \
    .setOutputCol("tokens")

embeddings = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

posology_ner_model = MedicalNerModel()\
    .pretrained("ner_posology_large", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols("sentences", "tokens", "ner")\
    .setOutputCol("ner_chunks")

pos_tager = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens")\
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

drug_chunk_embeddings = EntityChunkEmbeddings()\
    .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("drug_chunk_embeddings")\
    .setMaxSyntacticDistance(3)\
    .setTargetEntities({"DRUG": ["STRENGTH", "ROUTE", "FORM"]})\
    .setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2})

rxnorm_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")\
      .setInputCols(["drug_chunk_embeddings"])\
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_weighted_pipeline_re = Pipeline(
    stages = [
        documenter,
        sentence_detector,
        tokenizer,
        embeddings,
        posology_ner_model,
        ner_converter,
        pos_tager,
        dependency_parser,
        drug_chunk_embeddings,
        rxnorm_re
        ])
```
```scala
val documenter = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

val sentence_detector = SentenceDetector() \
    .setInputCols("documents") \
    .setOutputCol("sentences")

val tokenizer = Tokenizer() \
    .setInputCols("sentences") \
    .setOutputCol("tokens")

val embeddings = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(Array("sentences", "tokens"))\
    .setOutputCol("embeddings")

val posology_ner_model = MedicalNerModel()\
    .pretrained("ner_posology_large", "en", "clinical/models")\
    .setInputCols(Array("sentences", "tokens", "embeddings"))\
    .setOutputCol("ner")

val ner_converter = NerConverterInternal()\
    .setInputCols("sentences", "tokens", "ner")\
    .setOutputCol("ner_chunks")

val pos_tager = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens")\
    .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(Array("sentences", "pos_tags", "tokens"))\
    .setOutputCol("dependencies")

val drug_chunk_embeddings = EntityChunkEmbeddings()\
    .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(Array("ner_chunks", "dependencies"))\
    .setOutputCol("drug_chunk_embeddings")\
    .setMaxSyntacticDistance(3)\
    .setTargetEntities({"DRUG": ["STRENGTH", "ROUTE", "FORM"]})\
    .setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2})

val rxnorm_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")\
      .setInputCols(Array("drug_chunk_embeddings"))\
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_weighted_pipeline_re = new PipelineModel().setStages(Array(documenter, sentence_detector, tokenizer, embeddings, posology_ner_model, 
           ner_converter,  pos_tager, dependency_parser, drug_chunk_embeddings, rxnorm_re))

val light_model = LightPipeline(rxnorm_weighted_pipeline_re)

val result = light_model.fullAnnotate(Array("Coumadin 5 mg", "aspirin", ""avandia 4 mg"))
```
</div>

## Results

```bash
|    |   RxNormCode | Resolution                               | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-----------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]          | 855333:::432467:::438740:::103... | 3.0367:::4.7790:::4.7790:::5.3... | 0.0161:::0.0395:::0.0395:::0.0... | warfarin sodium 5 MG [Coumadin]:::coumarin 5 MG Oral Tablet:... | Branded Drug Comp:::Clinical D... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet         | 1537020:::1191:::1295740:::405... | 0.0000:::0.0000:::4.1826:::5.7... | 0.0000:::0.0000:::0.0292:::0.0... | aspirin Effervescent Oral Tablet:::aspirin:::aspirin Oral Po... | Clinical Drug Form:::Ingredien... |
|  2 |       261242 | rosiglitazone 4 MG Oral Tablet [Avandia] | 261242:::810073:::153845:::109... | 0.0000:::4.7482:::5.0125:::5.2... | 0.0000:::0.0365:::0.0409:::0.0... | rosiglitazone 4 MG Oral Tablet [Avandia]:::fesoterodine fuma... | Branded Drug:::Branded Drug Co... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_augmented_re|
|Compatibility:|Spark NLP for Healthcare 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|759.7 MB|
|Case sensitive:|false|