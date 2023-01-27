---
layout: model
title: ALBERT Large CoNNL-03 NER Pipeline
author: John Snow Labs
name: albert_large_token_classifier_conll03_pipeline
date: 2022-06-19
tags: [open_source, ner, token_classifier, albert, conll03, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [albert_large_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/09/26/albert_large_token_classifier_conll03_en.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_large_token_classifier_conll03_pipeline_en_4.0.0_3.0_1655653727302.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_large_token_classifier_conll03_pipeline_en_4.0.0_3.0_1655653727302.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


pipeline = PretrainedPipeline("albert_large_token_classifier_conll03_pipeline", lang = "en")

pipeline.annotate("My name is John and I work at John Snow Labs.")
```
```scala


val pipeline = new PretrainedPipeline("albert_large_token_classifier_conll03_pipeline", lang = "en")

pipeline.annotate("My name is John and I work at John Snow Labs."))
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
|Model Name:|albert_large_token_classifier_conll03_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|64.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- AlbertForTokenClassification
- NerConverter
- Finisher