---
layout: model
title: Malay (macrolanguage) malaysian_whisper_medium_pipeline pipeline WhisperForCTC from mesolitica
author: John Snow Labs
name: malaysian_whisper_medium_pipeline
date: 2024-12-17
tags: [ms, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ms
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malaysian_whisper_medium_pipeline` is a Malay (macrolanguage) model originally trained by mesolitica.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malaysian_whisper_medium_pipeline_ms_5.5.1_3.0_1734404159421.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malaysian_whisper_medium_pipeline_ms_5.5.1_3.0_1734404159421.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malaysian_whisper_medium_pipeline", lang = "ms")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malaysian_whisper_medium_pipeline", lang = "ms")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malaysian_whisper_medium_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ms|
|Size:|2.4 GB|

## References

https://huggingface.co/mesolitica/malaysian-whisper-medium

## Included Models

- AudioAssembler
- WhisperForCTC