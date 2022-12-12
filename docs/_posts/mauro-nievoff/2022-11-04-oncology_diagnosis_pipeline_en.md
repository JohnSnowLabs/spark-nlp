---
layout: model
title: Oncology Pipeline for Diagnosis Entities
author: John Snow Labs
name: oncology_diagnosis_pipeline
date: 2022-11-04
tags: [licensed, en, oncology, clinical]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.2.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline includes Named-Entity Recognition, Assertion Status, Relation Extraction and Entity Resolution models to extract information from oncology texts. This pipeline focuses on entities related to oncological diagnosis.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/oncology_diagnosis_pipeline_en_4.2.2_3.0_1667569522240.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("oncology_diagnosis_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("Two years ago, the patient presented with a 4-cm tumor in her left breast. She was diagnosed with ductal carcinoma. According to her last CT, she has no lung metastases.")[0]
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("oncology_diagnosis_pipeline", "en", "clinical/models")

val result = pipeline.fullAnnotate("""Two years ago, the patient presented with a 4-cm tumor in her left breast. She was diagnosed with ductal carcinoma. According to her last CT, she has no lung metastases.""")(0)
```
</div>

## Results

```bash
******************** ner_oncology_wip results ********************

| chunk      | ner_label         |
|:-----------|:------------------|
| 4-cm       | Tumor_Size        |
| tumor      | Tumor_Finding     |
| left       | Direction         |
| breast     | Site_Breast       |
| ductal     | Histological_Type |
| carcinoma  | Cancer_Dx         |
| lung       | Site_Lung         |
| metastases | Metastasis        |


******************** ner_oncology_diagnosis_wip results ********************

| chunk      | ner_label         |
|:-----------|:------------------|
| 4-cm       | Tumor_Size        |
| tumor      | Tumor_Finding     |
| ductal     | Histological_Type |
| carcinoma  | Cancer_Dx         |
| metastases | Metastasis        |


******************** ner_oncology_tnm_wip results ********************

| chunk      | ner_label         |
|:-----------|:------------------|
| 4-cm       | Tumor_Description |
| tumor      | Tumor             |
| ductal     | Tumor_Description |
| carcinoma  | Cancer_Dx         |
| metastases | Metastasis        |


******************** assertion_oncology_wip results ********************

| chunk      | ner_label         | assertion   |
|:-----------|:------------------|:------------|
| tumor      | Tumor_Finding     | Present     |
| ductal     | Histological_Type | Present     |
| carcinoma  | Cancer_Dx         | Present     |
| metastases | Metastasis        | Absent      |


******************** assertion_oncology_problem_wip results ********************

| chunk      | ner_label         | assertion              |
|:-----------|:------------------|:-----------------------|
| tumor      | Tumor_Finding     | Medical_History        |
| ductal     | Histological_Type | Medical_History        |
| carcinoma  | Cancer_Dx         | Medical_History        |
| metastases | Metastasis        | Hypothetical_Or_Absent |


******************** re_oncology_wip results ********************

| chunk1   | entity1       | chunk2     | entity2       | relation      |
|:---------|:--------------|:-----------|:--------------|:--------------|
| 4-cm     | Tumor_Size    | tumor      | Tumor_Finding | is_related_to |
| 4-cm     | Tumor_Size    | carcinoma  | Cancer_Dx     | O             |
| tumor    | Tumor_Finding | breast     | Site_Breast   | is_related_to |
| breast   | Site_Breast   | carcinoma  | Cancer_Dx     | O             |
| lung     | Site_Lung     | metastases | Metastasis    | is_related_to |


******************** re_oncology_granular_wip results ********************

| chunk1   | entity1       | chunk2     | entity2       | relation       |
|:---------|:--------------|:-----------|:--------------|:---------------|
| 4-cm     | Tumor_Size    | tumor      | Tumor_Finding | is_size_of     |
| 4-cm     | Tumor_Size    | carcinoma  | Cancer_Dx     | O              |
| tumor    | Tumor_Finding | breast     | Site_Breast   | is_location_of |
| breast   | Site_Breast   | carcinoma  | Cancer_Dx     | O              |
| lung     | Site_Lung     | metastases | Metastasis    | is_location_of |


******************** re_oncology_size_wip results ********************

| chunk1   | entity1    | chunk2    | entity2       | relation   |
|:---------|:-----------|:----------|:--------------|:-----------|
| 4-cm     | Tumor_Size | tumor     | Tumor_Finding | is_size_of |
| 4-cm     | Tumor_Size | carcinoma | Cancer_Dx     | O          |


******************** ICD-O resolver results ********************

| chunk      | ner_label         | code   | normalized_term   |
|:-----------|:------------------|:-------|:------------------|
| tumor      | Tumor_Finding     | 8000/1 | tumor             |
| breast     | Site_Breast       | C50    | breast            |
| ductal     | Histological_Type | 8500/2 | dcis              |
| carcinoma  | Cancer_Dx         | 8010/3 | carcinoma         |
| lung       | Site_Lung         | C34.9  | lung              |
| metastases | Metastasis        | 8000/6 | tumor, metastatic |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|oncology_diagnosis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|2.3 GB|

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
- ChunkMergeModel
- ChunkMergeModel
- AssertionDLModel
- AssertionDLModel
- PerceptronModel
- DependencyParserModel
- RelationExtractionModel
- RelationExtractionModel
- RelationExtractionModel
- ChunkMergeModel
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
