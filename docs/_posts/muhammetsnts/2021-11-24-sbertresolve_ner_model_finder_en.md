---
layout: model
title: NER Model Finder with Sentence Entity Resolvers (sbert_jsl_medium_uncased)
author: John Snow Labs
name: sbertresolve_ner_model_finder
date: 2021-11-24
tags: [ner, licensed, en, clinical, entity_resolver]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities (NER labels) to the most appropriate NER model using `sbert_jsl_medium_uncased` Sentence Bert Embeddings. Given the entity name, it will return a list of pretrained NER models having that entity or similar ones.

## Predicted Entities

`NER Model Names`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_ner_model_finder_en_3.3.2_2.4_1637764208798.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbertresolve_ner_model_finder_en_3.3.2_2.4_1637764208798.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")


sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbert_jsl_medium_uncased","en","clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")


ner_model_finder = SentenceEntityResolverModel\
    .pretrained("sbertresolve_ner_model_finder", "en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("model_names")\
    .setDistanceFunction("EUCLIDEAN")

    
ner_model_finder_pipelineModel = PipelineModel(stages = [documentAssembler, sbert_embedder, ner_model_finder])

light_pipeline = LightPipeline(ner_model_finder_pipelineModel)

annotations = light_pipeline.fullAnnotate("medication")

```
```scala
val documentAssembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
    .pretrained("sbert_jsl_medium_uncased","en","clinical/models")
    .setInputCols(Array("ner_chunk"))
    .setOutputCol("sbert_embeddings")
    
val ner_model_finder = SentenceEntityResolverModel
    .pretrained("sbertresolve_ner_model_finder", "en", "clinical/models")
    .setInputCols(Array("ner_chunk", "sbert_embeddings"))
    .setOutputCol("model_names")
    .setDistanceFunction("EUCLIDEAN")

    
val ner_model_finder_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, ner_model_finder))

val light_pipeline = LightPipeline(ner_model_finder_pipelineModel)

val annotations = light_pipeline.fullAnnotate("medication")

```
</div>

## Results

```bash
+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|entity    |models                                                                                                                                                                                                                                                                                                                                  |all_models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |resolutions                                                                                                                                                                                                                                                                                      |
+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|medication|['ner_posology', 'ner_posology_large', 'ner_posology_small', 'ner_posology_greedy', 'ner_drugs_large',  'ner_posology_experimental', 'ner_drugs_greedy', 'ner_ade_clinical', 'ner_jsl_slim', 'ner_posology_healthcare', 'ner_ade_healthcare', 'jsl_ner_wip_modifier_clinical', 'ner_ade_clinical', 'ner_jsl_greedy', 'ner_risk_factors']|['ner_posology', 'ner_posology_large', 'ner_posology_small', 'ner_posology_greedy', 'ner_drugs_large',  'ner_posology_experimental', 'ner_drugs_greedy', 'ner_ade_clinical', 'ner_jsl_slim', 'ner_posology_healthcare', 'ner_ade_healthcare', 'jsl_ner_wip_modifier_clinical', 'ner_ade_clinical', 'ner_jsl_greedy', 'ner_risk_factors']:::['ner_posology', 'ner_posology_large', 'ner_posology_small', 'ner_posology_greedy', 'ner_drugs_large',  'ner_posology_experimental', 'ner_drugs_greedy', 'ner_jsl_slim', 'ner_posology_healthcare', 'ner_ade_healthcare', 'jsl_ner_wip_modifier_clinical', 'ner_ade_clinical', 'ner_jsl_greedy', 'ner_risk_factors']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_clinical_large', 'ner_healthcare', 'ner_jsl_enriched', 'ner_clinical', 'ner_jsl_slim', 'ner_covid_trials', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_events_admission_clinical', 'ner_events_healthcare', 'ner_events_clinical', 'ner_jsl_greedy']:::['ner_medmentions_coarse']:::['ner_jsl_enriched', 'ner_covid_trials', 'ner_jsl', 'ner_medmentions_coarse']:::['ner_drugs']:::['ner_clinical_icdem', 'ner_medmentions_coarse']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_medmentions_coarse', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_medmentions_coarse', 'ner_radiology_wip_clinical', 'ner_jsl_slim', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy', 'ner_radiology']:::['ner_medmentions_coarse','ner_clinical_icdem']:::['ner_posology_experimental']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_measurements_clinical', 'ner_radiology_wip_clinical', 'ner_radiology']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_posology_small', 'ner_jsl_enriched', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_posology_greedy', 'ner_posology', 'ner_jsl_greedy']:::['ner_covid_trials', 'ner_medmentions_coarse', 'jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']:::['ner_deid_subentity_augmented', 'ner_deid_subentity_glove', 'ner_deidentify_dl', 'ner_deid_enriched']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_covid_trials', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']:::['ner_medmentions_coarse', 'jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']:::['ner_chemd_clinical']|medication:::drug:::treatment:::therapeutic procedure:::drug ingredient:::drug chemical:::diagnostic aid:::substance:::medical device:::diagnostic procedure:::administration:::measurement:::drug strength:::physiological reaction:::patient:::vaccine:::psychological condition:::abbreviation|
+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_ner_model_finder|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sbert_embeddings]|
|Output Labels:|[models]|
|Language:|en|
|Case sensitive:|false|
