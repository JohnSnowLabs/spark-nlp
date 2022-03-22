---
layout: model
title: Pipeline to Detect Problem, Test and Treatment
author: John Snow Labs
name: ner_healthcare_pipeline
date: 2022-03-22
tags: [licensed, ner, healthcare, treatment, problem, test, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_healthcare](https://nlp.johnsnowlabs.com/2021/04/21/ner_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_pipeline_en_3.4.1_3.0_1647943495587.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

pipeline.annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .")
```
```scala
val pipeline = new PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

pipeline.annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .")
```
</div>

## Results

```bash
|   | chunk                         | ner_label |
|---|-------------------------------|-----------|
| 0 | a respiratory tract infection | PROBLEM   |
| 1 | metformin                     | TREATMENT |
| 2 | glipizide                     | TREATMENT |
| 3 | dapagliflozin                 | TREATMENT |
| 4 | T2DM                          | PROBLEM   |
| 5 | atorvastatin                  | TREATMENT |
| 6 | gemfibrozil                   | TREATMENT |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.5 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter