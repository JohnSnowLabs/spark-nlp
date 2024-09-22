---
layout: model
title: Russian whisper_small_russian_v4_pipeline pipeline WhisperForCTC from sam-alavardo-1980
author: John Snow Labs
name: whisper_small_russian_v4_pipeline
date: 2024-09-21
tags: [ru, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_russian_v4_pipeline` is a Russian model originally trained by sam-alavardo-1980.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_russian_v4_pipeline_ru_5.5.0_3.0_1726893019344.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_russian_v4_pipeline_ru_5.5.0_3.0_1726893019344.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_russian_v4_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_russian_v4_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_russian_v4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.7 GB|

## References

https://huggingface.co/sam-alavardo-1980/whisper-small-ru-v4

## Included Models

- AudioAssembler
- WhisperForCTC