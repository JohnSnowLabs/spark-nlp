---
layout: model
title: Central Khmer, Khmer whisper_small_khmer_v2_pipeline pipeline WhisperForCTC from seanghay
author: John Snow Labs
name: whisper_small_khmer_v2_pipeline
date: 2024-09-06
tags: [km, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: km
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_khmer_v2_pipeline` is a Central Khmer, Khmer model originally trained by seanghay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_khmer_v2_pipeline_km_5.5.0_3.0_1725581546094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_khmer_v2_pipeline_km_5.5.0_3.0_1725581546094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_khmer_v2_pipeline", lang = "km")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_khmer_v2_pipeline", lang = "km")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_khmer_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|km|
|Size:|1.7 GB|

## References

https://huggingface.co/seanghay/whisper-small-khmer-v2

## Included Models

- AudioAssembler
- WhisperForCTC