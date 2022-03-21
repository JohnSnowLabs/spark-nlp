---
layout: model
title: Pipeline to Detect Adverse Drug Events (bert-clinical)
author: John Snow Labs
name: ner_ade_clinicalbert_pipeline
date: 2022-03-21
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

This pretrained pipeline is built on the top of [ner_ade_clinicalbert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinicalbert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_clinicalbert_pipeline_en_3.4.1_3.0_1647888358293.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_ade_clinicalbert_pipeline", "en", "clinical/models")


pipeline.annotate("EXAMPLE_TEXT")
```
```scala
val pipeline = new PretrainedPipeline("ner_ade_clinicalbert_pipeline", "en", "clinical/models")


pipeline.annotate("EXAMPLE_TEXT")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ade_clinicalbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter