---
layout: model
title: Castilian, Spanish maltese_hitz_spanish_basque_pipeline pipeline MarianTransformer from HiTZ
author: John Snow Labs
name: maltese_hitz_spanish_basque_pipeline
date: 2024-09-17
tags: [es, open_source, pipeline, onnx]
task: Translation
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`maltese_hitz_spanish_basque_pipeline` is a Castilian, Spanish model originally trained by HiTZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/maltese_hitz_spanish_basque_pipeline_es_5.5.0_3.0_1726581748715.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/maltese_hitz_spanish_basque_pipeline_es_5.5.0_3.0_1726581748715.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("maltese_hitz_spanish_basque_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("maltese_hitz_spanish_basque_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|maltese_hitz_spanish_basque_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|225.9 MB|

## References

https://huggingface.co/HiTZ/mt-hitz-es-eu

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer