---
layout: model
title: Bengali whisper_tiny_bengali_rakib_pipeline pipeline WhisperForCTC from Rakib
author: John Snow Labs
name: whisper_tiny_bengali_rakib_pipeline
date: 2024-09-16
tags: [bn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: bn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_bengali_rakib_pipeline` is a Bengali model originally trained by Rakib.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_bengali_rakib_pipeline_bn_5.5.0_3.0_1726480094832.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_bengali_rakib_pipeline_bn_5.5.0_3.0_1726480094832.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_bengali_rakib_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_bengali_rakib_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_bengali_rakib_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|391.3 MB|

## References

https://huggingface.co/Rakib/whisper-tiny-bn

## Included Models

- AudioAssembler
- WhisperForCTC