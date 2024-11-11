---
layout: model
title: Russian whisper_small_russian_f_pipeline pipeline WhisperForCTC from Garon16
author: John Snow Labs
name: whisper_small_russian_f_pipeline
date: 2024-11-11
tags: [ru, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_russian_f_pipeline` is a Russian model originally trained by Garon16.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_russian_f_pipeline_ru_5.5.1_3.0_1731304185329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_russian_f_pipeline_ru_5.5.1_3.0_1731304185329.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_russian_f_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_russian_f_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_russian_f_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.7 GB|

## References

https://huggingface.co/Garon16/whisper_small_ru_f

## Included Models

- AudioAssembler
- WhisperForCTC