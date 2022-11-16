---
layout: model
title: Longformer Large NER Pipeline
author: John Snow Labs
name: longformer_large_token_classifier_conll03_pipeline
date: 2022-04-20
tags: [open_source, ner, token_classifier, longformer, conll, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [longformer_large_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/10/09/longformer_large_token_classifier_conll03_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_large_token_classifier_conll03_pipeline_en_3.4.1_3.0_1650461801705.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("longformer_large_token_classifier_conll03_pipeline", lang = "en")

pipeline.annotate("My name is John and I am working at John Snow Labs.")
```
```scala
val pipeline = new PretrainedPipeline("longformer_large_token_classifier_conll03_pipeline", lang = "en")

pipeline.annotate("My name is John and I am working at John Snow Labs.")
```
</div>

## Results

```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|John          |PER      |
|John Snow Labs|ORG      |
+--------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longformer_large_token_classifier_conll03_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- LongformerForTokenClassification
- NerConverter
- Finisher