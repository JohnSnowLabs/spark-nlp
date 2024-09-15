---
layout: model
title: Persian persian_poem_classifier_pipeline pipeline AlbertForSequenceClassification from jrazi
author: John Snow Labs
name: persian_poem_classifier_pipeline
date: 2024-09-15
tags: [fa, open_source, pipeline, onnx]
task: Text Classification
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`persian_poem_classifier_pipeline` is a Persian model originally trained by jrazi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/persian_poem_classifier_pipeline_fa_5.5.0_3.0_1726372198442.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/persian_poem_classifier_pipeline_fa_5.5.0_3.0_1726372198442.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("persian_poem_classifier_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("persian_poem_classifier_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|persian_poem_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|44.1 MB|

## References

https://huggingface.co/jrazi/persian-poem-classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification