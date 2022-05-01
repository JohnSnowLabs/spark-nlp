---
layout: model
title: RE Pipeline between Dates and Clinical Entities
author: John Snow Labs
name: re_date_clinical_pipeline
date: 2022-03-31
tags: [licensed, clinical, relation_extraction, date, en]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [re_date_clinical](https://nlp.johnsnowlabs.com/2021/01/18/re_date_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_date_clinical_pipeline_en_3.4.1_3.0_1648734471721.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("re_date_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.")
```
```scala
val pipeline = new PretrainedPipeline("re_date_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.")
```
</div>

## Results

```bash
|   | relations | entity1 | entity1_begin | entity1_end | chunk1                                   | entity2 | entity2_end | entity2_end | chunk2  | confidence |
|---|-----------|---------|---------------|-------------|------------------------------------------|---------|-------------|-------------|---------|------------|
| 0 | 1         | Test    | 24            | 25          | CT                                       | Date    | 31          | 37          | 1/12/95 | 1.0        |
| 1 | 1         | Symptom | 45            | 84          | progressive memory and cognitive decline | Date    | 92          | 98          | 8/11/94 | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_date_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- PerceptronModel
- DependencyParserModel
- RelationExtractionModel