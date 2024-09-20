---
layout: model
title: Portuguese whisper_base_portuguese_koliskos_pipeline pipeline WhisperForCTC from koliskos
author: John Snow Labs
name: whisper_base_portuguese_koliskos_pipeline
date: 2024-09-14
tags: [pt, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_base_portuguese_koliskos_pipeline` is a Portuguese model originally trained by koliskos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_base_portuguese_koliskos_pipeline_pt_5.5.0_3.0_1726277860015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_base_portuguese_koliskos_pipeline_pt_5.5.0_3.0_1726277860015.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_base_portuguese_koliskos_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_base_portuguese_koliskos_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_base_portuguese_koliskos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|643.4 MB|

## References

https://huggingface.co/koliskos/whisper-base-pt

## Included Models

- AudioAssembler
- WhisperForCTC