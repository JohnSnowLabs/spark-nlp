---
layout: model
title: Hindi whisper_medium_hindi_shripadbhat_pipeline pipeline WhisperForCTC from shripadbhat
author: John Snow Labs
name: whisper_medium_hindi_shripadbhat_pipeline
date: 2024-09-21
tags: [hi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_hindi_shripadbhat_pipeline` is a Hindi model originally trained by shripadbhat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_hindi_shripadbhat_pipeline_hi_5.5.0_3.0_1726908022139.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_hindi_shripadbhat_pipeline_hi_5.5.0_3.0_1726908022139.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_hindi_shripadbhat_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_hindi_shripadbhat_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_hindi_shripadbhat_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|4.8 GB|

## References

https://huggingface.co/shripadbhat/whisper-medium-hi

## Included Models

- AudioAssembler
- WhisperForCTC