---
layout: model
title: Pipeline to Identify Adverse Drug Events
author: John Snow Labs
name: explain_clinical_doc_ade
date: 2021-02-11
task: Pipeline Healthcare
language: en
edition: Spark NLP 2.7.3
spark_version: 2.4
tags: [en, licensed]
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline containing multiple models to identify Adverse Drug Events in clinical and free text.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb#scrollTo=8i805kxSnnwA){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_2.7.3_2.4_1613049375392.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_2.7.3_2.4_1613049375392.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

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
| 1  | hydroxyurea                   | DRUG       | Present    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_ade|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

`biobert_pubmed_base_cased`, `classifierdl_ade_conversational_biobert`, `ner_ade_biobert` , `assertion_dl_biobert`