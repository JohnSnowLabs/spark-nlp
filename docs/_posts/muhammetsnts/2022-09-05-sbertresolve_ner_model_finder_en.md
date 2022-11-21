---
layout: model
title: NER Model Finder with Sentence Entity Resolvers (sbert_jsl_medium_uncased)
author: John Snow Labs
name: sbertresolve_ner_model_finder
date: 2022-09-05
tags: [en, entity_resolver, licensed, ner, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities (NER labels) to the most appropriate NER model using `sbert_jsl_medium_uncased` Sentence Bert Embeddings. Given the entity name, it will return a list of pretrained NER models having that entity or similar ones.

## Predicted Entities

`ner_model_list`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_ner_model_finder_en_4.1.0_3.0_1662377743401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_uncased","en","clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

ner_model_finder = SentenceEntityResolverModel.pretrained("sbertresolve_ner_model_finder", "en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("model_names")\
    .setDistanceFunction("EUCLIDEAN")

    
ner_model_finder_pipelineModel = PipelineModel(stages = [documentAssembler, sbert_embedder, ner_model_finder])

light_pipeline = LightPipeline(ner_model_finder_pipelineModel)

annotations = light_pipeline.fullAnnotate("medication")
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_uncased","en","clinical/models")
    .setInputCols(Array("ner_chunk"))
    .setOutputCol("sbert_embeddings")
    
val ner_model_finder = SentenceEntityResolverModel.pretrained("sbertresolve_ner_model_finder", "en", "clinical/models")
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
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|entity    |models                                                                                                                                                                                                                                                                                                                             |all_models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |resolutions                                                                                                                                                                                                                                                                       |
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|medication|['ner_posology_greedy', 'jsl_ner_wip_modifier_clinical', 'ner_posology_small', 'ner_jsl_greedy', 'ner_ade_clinical', 'ner_posology', 'ner_risk_factors', 'ner_ade_healthcare', 'ner_drugs_large', 'ner_jsl_slim', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare', 'ner_drugs_greedy', 'ner_pathogen']|['ner_posology_greedy', 'jsl_ner_wip_modifier_clinical', 'ner_posology_small', 'ner_jsl_greedy', 'ner_ade_clinical', 'ner_posology', 'ner_risk_factors', 'ner_ade_healthcare', 'ner_drugs_large', 'ner_jsl_slim', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare', 'ner_drugs_greedy', 'ner_pathogen']:::['ner_posology_greedy', 'jsl_ner_wip_modifier_clinical', 'ner_posology_small', 'ner_jsl_greedy', 'ner_ade_clinical', 'ner_nature_nero_clinical', 'ner_posology', 'ner_biomarker', 'ner_clinical_trials_abstracts', 'ner_risk_factors', 'ner_ade_healthcare', 'ner_drugs_large', 'ner_jsl_slim', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare', 'ner_drugs_greedy']:::['ner_covid_trials', 'ner_jsl', 'jsl_rd_ner_wip_greedy_clinical', 'jsl_ner_wip_modifier_clinical', 'ner_healthcare', 'ner_jsl_enriched', 'ner_events_clinical', 'ner_jsl_greedy', 'ner_clinical', 'ner_clinical_large', 'ner_jsl_slim', 'ner_events_healthcare', 'ner_events_admission_clinical']:::['ner_biomarker']:::['ner_medmentions_coarse']:::['ner_covid_trials', 'ner_jsl_enriched', 'ner_jsl', 'ner_medmentions_coarse']:::['ner_drugs']:::['ner_nature_nero_clinical']:::['ner_jsl', 'jsl_rd_ner_wip_greedy_clinical', 'jsl_ner_wip_modifier_clinical', 'ner_medmentions_coarse', 'ner_jsl_enriched', 'ner_jsl_greedy']:::['ner_jsl', 'jsl_rd_ner_wip_greedy_clinical', 'ner_nature_nero_clinical', 'ner_medmentions_coarse', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_enriched', 'ner_radiology_wip_clinical', 'ner_jsl_greedy', 'ner_radiology', 'ner_jsl_slim']:::['ner_posology_experimental']:::['ner_pathogen']:::['ner_measurements_clinical', 'jsl_rd_ner_wip_greedy_clinical', 'ner_nature_nero_clinical', 'ner_radiology_wip_clinical', 'ner_radiology', 'ner_nihss']:::['ner_jsl', 'ner_posology_greedy', 'jsl_rd_ner_wip_greedy_clinical', 'jsl_ner_wip_modifier_clinical', 'ner_posology_small', 'ner_jsl_enriched', 'ner_jsl_greedy', 'ner_posology', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare']:::['ner_covid_trials', 'ner_jsl', 'jsl_rd_ner_wip_greedy_clinical', 'ner_medmentions_coarse', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_enriched', 'ner_jsl_greedy']:::['ner_clinical_trials_abstracts']:::['ner_medmentions_coarse', 'ner_nature_nero_clinical']|medication:::drug:::treatment:::targeted therapy:::therapeutic procedure:::drug ingredient:::drug chemical:::medical procedure:::substance:::medical device:::administration:::medical condition:::measurement:::drug strength:::physiological reaction:::dose:::research activity|
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_ner_model_finder|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sbert_embeddings]|
|Output Labels:|[models]|
|Language:|en|
|Size:|737.3 KB|
|Case sensitive:|false|

## References

This model is trained with the data that has the labels of 70 different clinical NER models.