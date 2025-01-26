---
layout: model
title: Castilian, Spanish hatespeech_beto_cased_v1_pipeline pipeline BertForSequenceClassification from delarosajav95
author: John Snow Labs
name: hatespeech_beto_cased_v1_pipeline
date: 2025-01-25
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hatespeech_beto_cased_v1_pipeline` is a Castilian, Spanish model originally trained by delarosajav95.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hatespeech_beto_cased_v1_pipeline_es_5.5.1_3.0_1737801054087.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hatespeech_beto_cased_v1_pipeline_es_5.5.1_3.0_1737801054087.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hatespeech_beto_cased_v1_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hatespeech_beto_cased_v1_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hatespeech_beto_cased_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.7 MB|

## References

https://huggingface.co/delarosajav95/HateSpeech-BETO-cased-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification