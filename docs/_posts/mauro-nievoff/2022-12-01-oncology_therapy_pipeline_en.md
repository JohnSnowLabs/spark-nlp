---
layout: model
title: Oncology Pipeline for Therapies
author: John Snow Labs
name: oncology_therapy_pipeline
date: 2022-12-01
tags: [licensed, pipeline, oncology, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline includes Named-Entity Recognition and Assertion Status models to extract information from oncology texts. This pipeline focuses on entities related to therapies.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/oncology_therapy_pipeline_en_4.2.2_3.0_1669906146446.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/oncology_therapy_pipeline_en_4.2.2_3.0_1669906146446.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("oncology_therapy_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("The patient underwent a mastectomy two years ago. She is currently receiving her second cycle of adriamycin and cyclophosphamide, and is in good overall condition.")[0]

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("oncology_therapy_pipeline", "en", "clinical/models")

val result = pipeline.fullAnnotate("""The patient underwent a mastectomy two years ago. She is currently receiving her second cycle of adriamycin and cyclophosphamide, and is in good overall condition.""")(0)
```
</div>

## Results

```bash
******************** ner_oncology_wip results ********************

| chunk            | ner_label      |
|:-----------------|:---------------|
| mastectomy       | Cancer_Surgery |
| second cycle     | Cycle_Number   |
| adriamycin       | Chemotherapy   |
| cyclophosphamide | Chemotherapy   |


******************** ner_oncology_wip results ********************

| chunk            | ner_label      |
|:-----------------|:---------------|
| mastectomy       | Cancer_Surgery |
| second cycle     | Cycle_Number   |
| adriamycin       | Chemotherapy   |
| cyclophosphamide | Chemotherapy   |


******************** ner_oncology_wip results ********************

| chunk            | ner_label      |
|:-----------------|:---------------|
| mastectomy       | Cancer_Surgery |
| second cycle     | Cycle_Number   |
| adriamycin       | Cancer_Therapy |
| cyclophosphamide | Cancer_Therapy |


******************** ner_oncology_unspecific_posology_wip results ********************

| chunk            | ner_label            |
|:-----------------|:---------------------|
| mastectomy       | Cancer_Therapy       |
| second cycle     | Posology_Information |
| adriamycin       | Cancer_Therapy       |
| cyclophosphamide | Cancer_Therapy       |


******************** assertion_oncology_wip results ********************

| chunk            | ner_label      | assertion   |
|:-----------------|:---------------|:------------|
| mastectomy       | Cancer_Surgery | Past        |
| adriamycin       | Chemotherapy   | Present     |
| cyclophosphamide | Chemotherapy   | Present     |


******************** assertion_oncology_treatment_binary_wip results ********************

| chunk            | ner_label      | assertion       |
|:-----------------|:---------------|:----------------|
| mastectomy       | Cancer_Surgery | Present_Or_Past |
| adriamycin       | Chemotherapy   | Present_Or_Past |
| cyclophosphamide | Chemotherapy   | Present_Or_Past |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|oncology_therapy_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- MedicalNerModel
- NerConverter
- MedicalNerModel
- NerConverter
- MedicalNerModel
- NerConverter
- ChunkMergeModel
- ChunkMergeModel
- AssertionDLModel
- AssertionDLModel