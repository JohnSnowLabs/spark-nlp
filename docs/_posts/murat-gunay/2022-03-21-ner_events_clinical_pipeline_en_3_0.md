---
layout: model
title: Pipeline to Detect Clinical Events
author: John Snow Labs
name: ner_events_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
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

This pretrained pipeline is built on the top of [ner_events_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_events_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_pipeline_en_3.4.1_3.0_1647873847549.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_pipeline_en_3.4.1_3.0_1647873847549.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_events_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("The patient presented to the emergency room last evening")
```
```scala
val pipeline = new PretrainedPipeline("ner_events_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("The patient presented to the emergency room last evening")
```
</div>

## Results

```bash
+------------------+-------------+
|chunk             |ner_label    |
+------------------+-------------+
|presented         |OCCURRENCE   |
|the emergency room|CLINICAL_DEPT|
|last evening      |DATE         |
+------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_events_clinical_pipeline|
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
