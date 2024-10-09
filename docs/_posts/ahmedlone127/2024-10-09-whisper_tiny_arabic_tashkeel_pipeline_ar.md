---
layout: model
title: Arabic whisper_tiny_arabic_tashkeel_pipeline pipeline WhisperForCTC from WajeehAzeemX
author: John Snow Labs
name: whisper_tiny_arabic_tashkeel_pipeline
date: 2024-10-09
tags: [ar, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_arabic_tashkeel_pipeline` is a Arabic model originally trained by WajeehAzeemX.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_arabic_tashkeel_pipeline_ar_5.5.1_3.0_1728446056059.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_arabic_tashkeel_pipeline_ar_5.5.1_3.0_1728446056059.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_arabic_tashkeel_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_arabic_tashkeel_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_arabic_tashkeel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|390.0 MB|

## References

https://huggingface.co/WajeehAzeemX/whisper-tiny-ar-tashkeel

## Included Models

- AudioAssembler
- WhisperForCTC