---
layout: model
title: Pipeline for Adverse Drug Events
author: John Snow Labs
name: explain_clinical_doc_ade
date: 2021-04-01
tags: [pipeline, en, clinical, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline for Adverse Drug Events (ADE) with `ner_ade_biobert`, `assertiondl_biobert` and `classifierdl_ade_conversational_biobert`. It will extract ADE and DRUG clinical entities, assign assertion status to ADE entities, and then assign ADE status to a text (True means ADE, False means not related to ADE).

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_3.0.0_3.0_1617297946478.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_3.0.0_3.0_1617297946478.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

res = pipeline.fullAnnotate('The clinical course suggests that the interstitial pneumonitis was induced by hydroxyurea.')


```
```scala
val era_pipeline = new PretrainedPipeline("explain_clinical_doc_era", "en", "clinical/models")

val result = era_pipeline.fullAnnotate("""The clinical course suggests that the interstitial pneumonitis was induced by hydroxyurea.""")(0)

```
</div>

## Results

```bash
| #  | chunks                        | entities   | assertion  |
|----|-------------------------------|------------|------------|
| 0  | interstitial pneumonitis      | ADE        | Present    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_ade|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
- NerConverter
- AssertionDLModel
- SentenceEmbeddings
- ClassifierDLModel