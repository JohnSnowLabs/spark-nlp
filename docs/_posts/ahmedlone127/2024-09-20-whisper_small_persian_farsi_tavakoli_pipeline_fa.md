---
layout: model
title: Persian whisper_small_persian_farsi_tavakoli_pipeline pipeline WhisperForCTC from Tavakoli
author: John Snow Labs
name: whisper_small_persian_farsi_tavakoli_pipeline
date: 2024-09-20
tags: [fa, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_persian_farsi_tavakoli_pipeline` is a Persian model originally trained by Tavakoli.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_persian_farsi_tavakoli_pipeline_fa_5.5.0_3.0_1726876712959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_persian_farsi_tavakoli_pipeline_fa_5.5.0_3.0_1726876712959.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_persian_farsi_tavakoli_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_persian_farsi_tavakoli_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_persian_farsi_tavakoli_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|1.7 GB|

## References

https://huggingface.co/Tavakoli/whisper-small-fa

## Included Models

- AudioAssembler
- WhisperForCTC