---
layout: model
title: Serbian whisper_medium_serbian_v3_pipeline pipeline WhisperForCTC from Sagicc
author: John Snow Labs
name: whisper_medium_serbian_v3_pipeline
date: 2024-09-17
tags: [sr, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: sr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_serbian_v3_pipeline` is a Serbian model originally trained by Sagicc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_serbian_v3_pipeline_sr_5.5.0_3.0_1726548942733.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_serbian_v3_pipeline_sr_5.5.0_3.0_1726548942733.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_serbian_v3_pipeline", lang = "sr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_serbian_v3_pipeline", lang = "sr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_serbian_v3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sr|
|Size:|4.8 GB|

## References

https://huggingface.co/Sagicc/whisper-medium-sr-v3

## Included Models

- AudioAssembler
- WhisperForCTC