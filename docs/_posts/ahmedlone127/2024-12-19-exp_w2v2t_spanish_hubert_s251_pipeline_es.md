---
layout: model
title: Castilian, Spanish exp_w2v2t_spanish_hubert_s251_pipeline pipeline HubertForCTC from jonatasgrosman
author: John Snow Labs
name: exp_w2v2t_spanish_hubert_s251_pipeline
date: 2024-12-19
tags: [es, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`exp_w2v2t_spanish_hubert_s251_pipeline` is a Castilian, Spanish model originally trained by jonatasgrosman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/exp_w2v2t_spanish_hubert_s251_pipeline_es_5.5.1_3.0_1734599840885.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/exp_w2v2t_spanish_hubert_s251_pipeline_es_5.5.1_3.0_1734599840885.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("exp_w2v2t_spanish_hubert_s251_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("exp_w2v2t_spanish_hubert_s251_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|exp_w2v2t_spanish_hubert_s251_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|2.4 GB|

## References

https://huggingface.co/jonatasgrosman/exp_w2v2t_es_hubert_s251

## Included Models

- AudioAssembler
- HubertForCTC