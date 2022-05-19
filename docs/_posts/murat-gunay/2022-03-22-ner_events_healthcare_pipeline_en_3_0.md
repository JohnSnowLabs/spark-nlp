---
layout: model
title: Pipeline to Detect Clinical Events
author: John Snow Labs
name: ner_events_healthcare_pipeline
date: 2022-03-22
tags: [licensed, ner, clinical, en]
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

This pretrained pipeline is built on the top of [ner_events_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_pipeline_en_3.4.1_3.0_1647943997404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")

pipeline.annotate("The patient presented to the emergency room last evening")
```
```scala
val pipeline = new PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")

pipeline.annotate("The patient presented to the emergency room last evening")
```
</div>

## Results

```bash
+------------------+-------------+
|chunks            |entities     |
+------------------+-------------+
|presented         |EVIDENTIAL   |
|the emergency room|CLINICAL_DEPT|
|last evening      |DATE         |
+------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_events_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
