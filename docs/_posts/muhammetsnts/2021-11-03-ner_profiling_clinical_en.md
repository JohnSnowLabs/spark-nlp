---
layout: model
title: Named Entity Recognition Profiling (Clinical)
author: John Snow Labs
name: ner_profiling_clinical
date: 2021-11-03
tags: [ner, ner_profiling, clinical, en, licensed]
task: Pipeline Healthcare
language: en
edition: Spark NLP for Healthcare 3.3.1
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline can be used to explore all the available pretrained NER models at once. When you run this pipeline over your text, you will end up with the predictions coming out of each pretrained clinical NER model trained with `embeddings_clinical`. It has been updated by adding new clinical NER model and NER model outputs to the previous version.

Here are the NER models that this pretrained pipeline includes: `ner_ade_clinical`, `ner_posology_greedy`, `ner_risk_factors`, `jsl_ner_wip_clinical`, `ner_human_phenotype_gene_clinical`, `jsl_ner_wip_greedy_clinical`, `ner_cellular`, `ner_cancer_genetics`, `jsl_ner_wip_modifier_clinical`, `ner_drugs_greedy`, `ner_deid_sd_large`, `ner_diseases`, `nerdl_tumour_demo`, `ner_deid_subentity_augmented`, `ner_jsl_enriched`, `ner_genetic_variants`, `ner_bionlp`, `ner_measurements_clinical`, `ner_diseases_large`, `ner_radiology`, `ner_deid_augmented`, `ner_anatomy`, `ner_chemprot_clinical`, `ner_posology_experimental`, `ner_drugs`, `ner_deid_sd`, `ner_posology_large`, `ner_deid_large`, `ner_posology`, `ner_deidentify_dl`, `ner_deid_enriched`, `ner_bacterial_species`, `ner_drugs_large`, `ner_clinical_large`, `jsl_rd_ner_wip_greedy_clinical`, `ner_medmentions_coarse`, `ner_radiology_wip_clinical`, `ner_clinical`, `ner_chemicals`, `ner_deid_synthetic`, `ner_events_clinical`, `ner_posology_small`, `ner_anatomy_coarse`, `ner_human_phenotype_go_clinical`, `ner_jsl_slim`, `ner_jsl`, `ner_jsl_greedy`, `ner_events_admission_clinical`, `ner_chexpert` .

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_profiling_clinical_en_3.3.1_2.4_1635974903665.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
</div>

## Results

```bash
 ******************** ner_jsl Model Results ******************** 

[('28-year-old', 'Age'), ('female', 'Gender'), ('gestational diabetes mellitus', 'Diabetes'), ('eight years prior', 'RelativeDate'), ('subsequent', 'Modifier'), ('type two diabetes mellitus', 'Diabetes'), ('T2DM', 'Diabetes'), ('HTG-induced pancreatitis', 'Disease_Syndrome_Disorder'), ('three years prior', 'RelativeDate'), ('acute', 'Modifier'), ('hepatitis', 'Communicable_Disease'), ('obesity', 'Obesity'), ('body mass index', 'Symptom'), ('33.5 kg/m2', 'Weight'), ('one-week', 'Duration'), ('polyuria', 'Symptom'), ('polydipsia', 'Symptom'), ('poor appetite', 'Symptom'), ('vomiting', 'Symptom')]


 ******************** ner_diseases_large Model Results ******************** 

[('gestational diabetes mellitus', 'Disease'), ('diabetes mellitus', 'Disease'), ('T2DM', 'Disease'), ('pancreatitis', 'Disease'), ('hepatitis', 'Disease'), ('obesity', 'Disease'), ('polyuria', 'Disease'), ('polydipsia', 'Disease'), ('vomiting', 'Disease')]

 
 ******************** ner_radiology Model Results ******************** 

[('gestational diabetes mellitus', 'Disease_Syndrome_Disorder'), ('type two diabetes mellitus', 'Disease_Syndrome_Disorder'), ('T2DM', 'Disease_Syndrome_Disorder'), ('HTG-induced pancreatitis', 'Disease_Syndrome_Disorder'), ('acute hepatitis', 'Disease_Syndrome_Disorder'), ('obesity', 'Disease_Syndrome_Disorder'), ('body', 'BodyPart'), ('mass index', 'Symptom'), ('BMI', 'Test'), ('33.5', 'Measurements'), ('kg/m2', 'Units'), ('polyuria', 'Symptom'), ('polydipsia', 'Symptom'), ('poor appetite', 'Symptom'), ('vomiting', 'Symptom')]


 ******************** ner_clinical Model Results ******************** 

[('gestational diabetes mellitus', 'PROBLEM'), ('subsequent type two diabetes mellitus', 'PROBLEM'), ('T2DM', 'PROBLEM'), ('HTG-induced pancreatitis', 'PROBLEM'), ('an acute hepatitis', 'PROBLEM'), ('obesity', 'PROBLEM'), ('a body mass index', 'PROBLEM'), ('BMI', 'TEST'), ('polyuria', 'PROBLEM'), ('polydipsia', 'PROBLEM'), ('poor appetite', 'PROBLEM'), ('vomiting', 'PROBLEM')]


 ******************** ner_medmentions_coarse Model Results ******************** 

[('female', 'Organism_Attribute'), ('diabetes mellitus', 'Disease_or_Syndrome'), ('diabetes mellitus', 'Disease_or_Syndrome'), ('T2DM', 'Disease_or_Syndrome'), ('HTG-induced pancreatitis', 'Disease_or_Syndrome'), ('associated with', 'Qualitative_Concept'), ('acute hepatitis', 'Disease_or_Syndrome'), ('obesity', 'Disease_or_Syndrome'), ('body mass index', 'Clinical_Attribute'), ('BMI', 'Clinical_Attribute'), ('polyuria', 'Sign_or_Symptom'), ('polydipsia', 'Sign_or_Symptom'), ('poor appetite', 'Sign_or_Symptom'), ('vomiting', 'Sign_or_Symptom')]

...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_profiling_clinical|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.3.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel (x49)
- NerConverter (x49)
- Finisher
