---
layout: model
title: Oriya (macrolanguage) whisper_cli_dropout_small_oriya_pipeline pipeline WhisperForCTC from auro
author: John Snow Labs
name: whisper_cli_dropout_small_oriya_pipeline
date: 2024-09-17
tags: [or, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: or
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_cli_dropout_small_oriya_pipeline` is a Oriya (macrolanguage) model originally trained by auro.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_cli_dropout_small_oriya_pipeline_or_5.5.0_3.0_1726546430139.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_cli_dropout_small_oriya_pipeline_or_5.5.0_3.0_1726546430139.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_cli_dropout_small_oriya_pipeline", lang = "or")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_cli_dropout_small_oriya_pipeline", lang = "or")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_cli_dropout_small_oriya_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|or|
|Size:|1.7 GB|

## References

https://huggingface.co/auro/whisper-cli-dropout-small-or

## Included Models

- AudioAssembler
- WhisperForCTC