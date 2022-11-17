---
layout: model
title: Pipeline for Adverse Drug Events
author: John Snow Labs
name: explain_clinical_doc_ade
date: 2022-06-30
tags: [en, clinical, licensed, ade, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline for Adverse Drug Events (ADE) with `ner_ade_biobert`, `assertion_dl_biobert`,  ,`classifierdl_ade_conversational_biobert`, and `re_ade_biobert` . It will classify the document, extract ADE and DRUG clinical entities, assign assertion status to ADE entities, and relate Drugs with their ADEs.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_4.0.0_3.0_1656581944018.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("explain_clinical_doc_ade", "en", "clinical/models")

res = pipeline.fullAnnotate("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")
```
```scala
from sparknlp.pretrained import PretrainedPipeline

val era_pipeline = new PretrainedPipeline("explain_clinical_doc_ade", "en", "clinical/models")

val result = era_pipeline.fullAnnotate("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")(0)

```
</div>

## Results

```bash
Class: True

NER_Assertion:
|    | chunk                   | entitiy    | assertion   |
|----|-------------------------|------------|-------------|
| 0  | Lipitor                 | DRUG       | -           |
| 1  | severe fatigue          | ADE        | Conditional |
| 2  | voltaren                | DRUG       | -           |
| 3  | cramps                  | ADE        | Conditional |

Relations:
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation |
|----|-------------------------------|------------|-------------|---------|----------|
| 0  | severe fatigue                | ADE        | Lipitor     | DRUG    |        1 |
| 1  | cramps                        | ADE        | Lipitor     | DRUG    |        0 |
| 2  | severe fatigue                | ADE        | voltaren    | DRUG    |        0 |
| 3  | cramps                        | ADE        | voltaren    | DRUG    |        1 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_ade|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|484.6 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings
- SentenceEmbeddings
- ClassifierDLModel
- MedicalNerModel
- NerConverterInternalModel
- PerceptronModel
- DependencyParserModel
- RelationExtractionModel
- NerConverterInternalModel
- AssertionDLModel
