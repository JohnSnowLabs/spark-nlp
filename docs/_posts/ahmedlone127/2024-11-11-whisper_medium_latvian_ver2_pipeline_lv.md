---
layout: model
title: Latvian whisper_medium_latvian_ver2_pipeline pipeline WhisperForCTC from FelixK7
author: John Snow Labs
name: whisper_medium_latvian_ver2_pipeline
date: 2024-11-11
tags: [lv, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: lv
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_latvian_ver2_pipeline` is a Latvian model originally trained by FelixK7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_latvian_ver2_pipeline_lv_5.5.1_3.0_1731305454634.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_latvian_ver2_pipeline_lv_5.5.1_3.0_1731305454634.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_latvian_ver2_pipeline", lang = "lv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_latvian_ver2_pipeline", lang = "lv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_latvian_ver2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|lv|
|Size:|4.8 GB|

## References

https://huggingface.co/FelixK7/whisper-medium-lv-ver2

## Included Models

- AudioAssembler
- WhisperForCTC