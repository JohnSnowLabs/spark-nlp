---
layout: model
title: Multilingual aisak_listen_pipeline pipeline WhisperForCTC from aisak-ai
author: John Snow Labs
name: aisak_listen_pipeline
date: 2024-09-19
tags: [xx, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`aisak_listen_pipeline` is a Multilingual model originally trained by aisak-ai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/aisak_listen_pipeline_xx_5.5.0_3.0_1726714390588.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/aisak_listen_pipeline_xx_5.5.0_3.0_1726714390588.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("aisak_listen_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("aisak_listen_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|aisak_listen_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|242.9 MB|

## References

https://huggingface.co/aisak-ai/aisak-listen

## Included Models

- AudioAssembler
- WhisperForCTC