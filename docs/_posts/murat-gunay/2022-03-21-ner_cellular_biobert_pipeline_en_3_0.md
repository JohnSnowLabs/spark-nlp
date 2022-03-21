---
layout: model
title: Pipeline to Detect details of cellular structures (biobert)
author: John Snow Labs
name: ner_cellular_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_cellular_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_cellular_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cellular_biobert_pipeline_en_3.4.1_3.0_1647870475485.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_cellular_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("EXAMPLE_TEXT")
```
```scala
val pipeline = new PretrainedPipeline("ner_cellular_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("EXAMPLE_TEXT")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_cellular_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|421.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter