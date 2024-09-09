---
layout: model
title: Welsh whisper_tiny_ft_welsh_english_pipeline pipeline WhisperForCTC from techiaith
author: John Snow Labs
name: whisper_tiny_ft_welsh_english_pipeline
date: 2024-09-07
tags: [cy, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: cy
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_ft_welsh_english_pipeline` is a Welsh model originally trained by techiaith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_ft_welsh_english_pipeline_cy_5.5.0_3.0_1725752104784.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_ft_welsh_english_pipeline_cy_5.5.0_3.0_1725752104784.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_ft_welsh_english_pipeline", lang = "cy")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_ft_welsh_english_pipeline", lang = "cy")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_ft_welsh_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|cy|
|Size:|389.6 MB|

## References

https://huggingface.co/techiaith/whisper-tiny-ft-cy-en

## Included Models

- AudioAssembler
- WhisperForCTC