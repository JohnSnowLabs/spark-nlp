---
layout: model
title: Named Entity Recognition Profiling (Clinical)
author: John Snow Labs
name: ner_profiling_clinical
date: 2021-09-24
tags: [ner, ner_profiling, clinical, licensed, en]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.2.3
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline can be used to explore all the available pretrained NER models at once. When you run this pipeline over your text, you will end up with the predictions coming out of each pretrained clinical NER model trained with `embeddings_clinical`.

Here are the NER models that this pretrained pipeline includes: `ner_ade_clinical_chunks`, `ner_posology_greedy_chunks`, `ner_risk_factors_chunks`, `jsl_ner_wip_clinical_chunks`, `ner_human_phenotype_gene_clinical_chunks`, `jsl_ner_wip_greedy_clinical_chunks`, `ner_cellular_chunks`, `ner_cancer_genetics_chunks`, `jsl_ner_wip_modifier_clinical_chunks`, `ner_drugs_greedy_chunks`, `ner_deid_sd_large_chunks`, `ner_diseases_chunks`, `nerdl_tumour_demo_chunks`, `ner_deid_subentity_augmented_chunks`, `ner_jsl_enriched_chunks`, `ner_genetic_variants_chunks`, `ner_bionlp_chunks`, `ner_measurements_clinical_chunks`, `ner_diseases_large_chunks`, `ner_radiology_chunks`, `ner_deid_augmented_chunks`, `ner_anatomy_chunks`, `ner_chemprot_clinical_chunks`, `ner_posology_experimental_chunks`, `ner_drugs_chunks`, `ner_deid_sd_chunks`, `ner_posology_large_chunks`, `ner_deid_large_chunks`, `ner_posology_chunks`, `ner_deidentify_dl_chunks`, `ner_deid_enriched_chunks`, `ner_bacterial_species_chunks`, `ner_drugs_large_chunks`, `ner_clinical_large_chunks`, `jsl_rd_ner_wip_greedy_clinical_chunks`, `ner_medmentions_coarse_chunks`, `ner_radiology_wip_clinical_chunks`, `ner_clinical_chunks`, `ner_chemicals_chunks`, `ner_deid_synthetic_chunks`, `ner_events_clinical_chunks`, `ner_posology_small_chunks`, `ner_anatomy_coarse_chunks`, `ner_human_phenotype_go_clinical_chunks`, `ner_jsl_slim_chunks`, `ner_jsl_chunks`, `ner_jsl_greedy_chunks`, `ner_events_admission_clinical_chunks` .

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_profiling_clinical_en_3.2.3_2.4_1632491778580.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_profiling_clinical_en_3.2.3_2.4_1632491778580.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

ner_profiling_pipeline = PretrainedPipeline('ner_profiling_clinical', 'en', 'clinical/models')

result = ner_profiling_pipeline.annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val ner_profiling_pipeline = PretrainedPipeline('ner_profiling_clinical', 'en', 'clinical/models')

val result = ner_profiling_pipeline.annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.profiling_clinical").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .""")
```

</div>

## Results

```bash
ner_ade_clinical_chunks :  ['polydipsia', 'poor appetite', 'vomiting']
ner_posology_greedy_chunks :  []
ner_risk_factors_chunks :  ['diabetes mellitus', 'type two diabetes mellitus', 'obesity']
jsl_ner_wip_clinical_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'subsequent', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute', 'hepatitis', 'obesity', 'body mass index', '33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_human_phenotype_gene_clinical_chunks :  ['type', 'obesity', 'mass', 'polyuria', 'polydipsia']
jsl_ner_wip_greedy_clinical_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_cellular_chunks :  []
ner_cancer_genetics_chunks :  []
jsl_ner_wip_modifier_clinical_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_drugs_greedy_chunks :  []
ner_deid_sd_large_chunks :  []
ner_diseases_chunks :  ['gestational diabetes mellitus', 'diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'hepatitis', 'obesity', 'BMI', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
nerdl_tumour_demo_chunks :  []
ner_deid_subentity_augmented_chunks :  ['28-year-old']
ner_jsl_enriched_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'acute', 'hepatitis', 'obesity', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_genetic_variants_chunks :  []
ner_bionlp_chunks :  ['female', 'hepatitis']
ner_measurements_clinical_chunks :  ['33.5', 'kg/m2']
ner_diseases_large_chunks :  ['gestational diabetes mellitus', 'diabetes mellitus', 'T2DM', 'pancreatitis', 'hepatitis', 'obesity', 'polyuria', 'polydipsia', 'vomiting']
ner_radiology_chunks :  ['gestational diabetes mellitus', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'acute hepatitis', 'obesity', 'body', 'mass index', 'BMI', '33.5', 'kg/m2', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_deid_augmented_chunks :  []
ner_anatomy_chunks :  ['body']
ner_chemprot_clinical_chunks :  []
ner_posology_experimental_chunks :  []
ner_drugs_chunks :  []
ner_deid_sd_chunks :  []
ner_posology_large_chunks :  []
ner_deid_large_chunks :  []
ner_posology_chunks :  []
ner_deidentify_dl_chunks :  []
ner_deid_enriched_chunks :  []
ner_bacterial_species_chunks :  []
ner_drugs_large_chunks :  []
ner_clinical_large_chunks :  ['gestational diabetes mellitus', 'subsequent type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'an acute hepatitis', 'obesity', 'a body mass index', 'BMI', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
token :  ['A', '28-year-old', 'female', 'with', 'a', 'history', 'of', 'gestational', 'diabetes', 'mellitus', 'diagnosed', 'eight', 'years', 'prior', 'to', 'presentation', 'and', 'subsequent', 'type', 'two', 'diabetes', 'mellitus', '(', 'T2DM', '),', 'one', 'prior', 'episode', 'of', 'HTG-induced', 'pancreatitis', 'three', 'years', 'prior', 'to', 'presentation', ',', 'associated', 'with', 'an', 'acute', 'hepatitis', ',', 'and', 'obesity', 'with', 'a', 'body', 'mass', 'index', '(', 'BMI', ')', 'of', '33.5', 'kg/m2', ',', 'presented', 'with', 'a', 'one-week', 'history', 'of', 'polyuria', ',', 'polydipsia', ',', 'poor', 'appetite', ',', 'and', 'vomiting', '.']
jsl_rd_ner_wip_greedy_clinical_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'subsequent type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass index ( BMI', '33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_medmentions_coarse_chunks :  ['female', 'diabetes mellitus', 'diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'associated with', 'acute hepatitis', 'obesity', 'body mass index', 'BMI', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_radiology_wip_clinical_chunks :  ['gestational diabetes mellitus', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'acute hepatitis', 'obesity', 'body', 'mass index', '33.5', 'kg/m2', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_clinical_chunks :  ['gestational diabetes mellitus', 'subsequent type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'an acute hepatitis', 'obesity', 'a body mass index', 'BMI', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_chemicals_chunks :  []
ner_deid_synthetic_chunks :  []
ner_events_clinical_chunks :  ['gestational diabetes mellitus', 'eight years', 'presentation', 'subsequent type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years', 'presentation', 'an acute hepatitis', 'obesity', 'a body mass index ( BMI', 'presented', 'a one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_posology_small_chunks :  []
ner_anatomy_coarse_chunks :  ['body']
ner_human_phenotype_go_clinical_chunks :  ['obesity', 'polydipsia', 'vomiting']
ner_jsl_slim_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass index', 'BMI ) of 33.5 kg/m2', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_jsl_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'subsequent', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute', 'hepatitis', 'obesity', 'body mass index', '33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_jsl_greedy_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
sentence :  ['A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .']
ner_events_admission_clinical_chunks :  ['gestational diabetes mellitus', 'eight years', 'presentation', 'subsequent type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years', 'presentation', 'an acute hepatitis', 'obesity', 'a body mass index', 'BMI', 'kg/m2', 'presented', 'a one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_profiling_clinical|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel (x48)
- NerConverter (x48)
- Finisher
