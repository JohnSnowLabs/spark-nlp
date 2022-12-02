---
layout: model
title: Pipeline to Detect Medication Entities, Assign Assertion Status and Find Relations
author: John Snow Labs
name: explain_clinical_doc_medication
date: 2022-04-01
tags: [licensed, en, clinical, ner, assertion, relation_extraction, posology]
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

A pipeline for detecting posology entities with the `ner_posology_large` NER model, assigning their assertion status with `assertion_jsl` model, and extracting relations between posology-related terminology with `posology_re` relation extraction model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_medication_en_3.4.2_3.0_1648813363898.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("explain_clinical_doc_medication", "en", "clinical/models")

result = pipeline.fullAnnotate("""The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2. She received a course of Bactrim for 14 days for UTI.  She was prescribed 5000 units of Fragmin  subcutaneously daily, and along with Lantus 40 units subcutaneously at bedtime.""")[0]

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("explain_clinical_doc_medication", "en", "clinical/models")

val result = pipeline.fullAnnotate("""The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2. She received a course of Bactrim for 14 days for UTI.  She was prescribed 5000 units of Fragmin  subcutaneously daily, and along with Lantus 40 units subcutaneously at bedtime.""")(0)

```
</div>

## Results

```bash
+----+----------------+------------+
|    | chunks         | entities   |
|---:|:---------------|:-----------|
|  0 | insulin        | DRUG       |
|  1 | Bactrim        | DRUG       |
|  2 | for 14 days    | DURATION   |
|  3 | 5000 units     | DOSAGE     |
|  4 | Fragmin        | DRUG       |
|  5 | subcutaneously | ROUTE      |
|  6 | daily          | FREQUENCY  |
|  7 | Lantus         | DRUG       |
|  8 | 40 units       | DOSAGE     |
|  9 | subcutaneously | ROUTE      |
| 10 | at bedtime     | FREQUENCY  |
+----+----------------+------------+

+----+----------+------------+-------------+
|    | chunks   | entities   | assertion   |
|---:|:---------|:-----------|:------------|
|  0 | insulin  | DRUG       | Present     |
|  1 | Bactrim  | DRUG       | Past        |
|  2 | Fragmin  | DRUG       | Planned     |
|  3 | Lantus   | DRUG       | Planned     |
+----+----------+------------+-------------+

+----------------+-----------+------------+-----------+----------------+
| relation       | entity1   | chunk1     | entity2   | chunk2         |
|:---------------|:----------|:-----------|:----------|:---------------|
| DRUG-DURATION  | DRUG      | Bactrim    | DURATION  | for 14 days    |
| DOSAGE-DRUG    | DOSAGE    | 5000 units | DRUG      | Fragmin        |
| DRUG-ROUTE     | DRUG      | Fragmin    | ROUTE     | subcutaneously |
| DRUG-FREQUENCY | DRUG      | Fragmin    | FREQUENCY | daily          |
| DRUG-DOSAGE    | DRUG      | Lantus     | DOSAGE    | 40 units       |
| DRUG-ROUTE     | DRUG      | Lantus     | ROUTE     | subcutaneously |
| DRUG-FREQUENCY | DRUG      | Lantus     | FREQUENCY | at bedtime     |
+----------------+-----------+------------+-----------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_medication|
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
- PosologyREModel