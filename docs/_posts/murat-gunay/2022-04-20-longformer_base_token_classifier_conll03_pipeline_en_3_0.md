---
layout: model
title: Longformer Base NER Pipeline
author: John Snow Labs
name: longformer_base_token_classifier_conll03_pipeline
date: 2022-04-20
tags: [ner, longformer, pipeline, conll, token_classification, en, open_source]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [longformer_base_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/10/09/longformer_base_token_classifier_conll03_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_base_token_classifier_conll03_pipeline_en_3.4.1_3.0_1650456150982.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/longformer_base_token_classifier_conll03_pipeline_en_3.4.1_3.0_1650456150982.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("longformer_base_token_classifier_conll03_pipeline", lang = "en")

pipeline.annotate("My name is John and I am working at John Snow Labs.")
```
```scala
val pipeline = new PretrainedPipeline("longformer_base_token_classifier_conll03_pipeline", lang = "en")

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
|Model Name:|longformer_base_token_classifier_conll03_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|516.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- LongformerForTokenClassification
- NerConverter
- Finisher