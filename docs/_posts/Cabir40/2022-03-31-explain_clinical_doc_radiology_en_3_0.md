---
layout: model
title: Pipeline to Detect Radiology Entities, Assign Assertion Status and Find Relations
author: John Snow Labs
name: explain_clinical_doc_radiology
date: 2022-03-31
tags: [licensed, clinical, en, ner, assertion, relation_extraction, radiology]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline for detecting radiology entities with the `ner_radiology` NER model, assigning their assertion status with `assertion_dl_radiology` model, and extracting relations between the diagnosis, test, and findings with `re_test_problem_finding` relation extraction model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_radiology_en_3.4.2_3.0_1648737971620.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("explain_clinical_doc_radiology", "en", "clinical/models")

result = pipeline.fullAnnotate("""Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.""")[0]

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("explain_clinical_doc_radiology", "en", "clinical/models")

val result = pipeline.fullAnnotate("""Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.""")(0)

```
</div>

## Results

```bash
+----+------------------------------------------+---------------------------+
|    | chunks                                   | entities                  |
|---:|:-----------------------------------------|:--------------------------|
|  0 | Bilateral breast                         | BodyPart                  |
|  1 | ultrasound                               | ImagingTest               |
|  2 | ovoid mass                               | ImagingFindings           |
|  3 | 0.5 x 0.5 x 0.4                          | Measurements              |
|  4 | cm                                       | Units                     |
|  5 | anteromedial aspect of the left shoulder | BodyPart                  |
|  6 | mass                                     | ImagingFindings           |
|  7 | isoechoic echotexture                    | ImagingFindings           |
|  8 | muscle                                   | BodyPart                  |
|  9 | internal color flow                      | ImagingFindings           |
| 10 | benign fibrous tissue                    | ImagingFindings           |
| 11 | lipoma                                   | Disease_Syndrome_Disorder |
+----+------------------------------------------+---------------------------+

+----+-----------------------+---------------------------+-------------+
|    | chunks                | entities                  | assertion   |
|---:|:----------------------|:--------------------------|:------------|
|  0 | ultrasound            | ImagingTest               | Confirmed   |
|  1 | ovoid mass            | ImagingFindings           | Confirmed   |
|  2 | mass                  | ImagingFindings           | Confirmed   |
|  3 | isoechoic echotexture | ImagingFindings           | Confirmed   |
|  4 | internal color flow   | ImagingFindings           | Negative    |
|  5 | benign fibrous tissue | ImagingFindings           | Suspected   |
|  6 | lipoma                | Disease_Syndrome_Disorder | Suspected   |
+----+-----------------------+---------------------------+-------------+

+---------+-----------------+-----------------------+---------------------------+------------+
|relation | entity1         | chunk1                | entity2                   | chunk2     |
|--------:|:----------------|:----------------------|:--------------------------|:-----------|
|       1 | ImagingTest     | ultrasound            | ImagingFindings           | ovoid mass |
|       0 | ImagingFindings | benign fibrous tissue | Disease_Syndrome_Disorder | lipoma     |
+---------+-----------------+-----------------------+---------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_radiology|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.2+|
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
- NerConverterInternal
- NerConverterInternal
- AssertionDLModel
- PerceptronModel
- DependencyParserModel
- RelationExtractionModel
