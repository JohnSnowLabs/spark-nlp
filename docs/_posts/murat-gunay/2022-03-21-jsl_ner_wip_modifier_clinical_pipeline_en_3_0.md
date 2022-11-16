---
layout: model
title: Pipeline to Detect Clinical Concepts (WIP Modifier)
author: John Snow Labs
name: jsl_ner_wip_modifier_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, wip, clinical, modifier, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [jsl_ner_wip_modifier_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_ner_wip_modifier_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_modifier_clinical_pipeline_en_3.4.1_3.0_1647866811633.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("jsl_ner_wip_modifier_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("EXAMPLE MEDICAL TEXT")
```
```scala
val pipeline = new PretrainedPipeline("jsl_ner_wip_modifier_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("EXAMPLE MEDICAL TEXT")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_modifier_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
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