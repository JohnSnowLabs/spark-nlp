---
layout: model
title: Dhivehi, Divehi, Maldivian whisper_small_divehi_bochen0909_pipeline pipeline WhisperForCTC from bochen0909
author: John Snow Labs
name: whisper_small_divehi_bochen0909_pipeline
date: 2024-10-09
tags: [dv, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: dv
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_divehi_bochen0909_pipeline` is a Dhivehi, Divehi, Maldivian model originally trained by bochen0909.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_divehi_bochen0909_pipeline_dv_5.5.1_3.0_1728444727807.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_divehi_bochen0909_pipeline_dv_5.5.1_3.0_1728444727807.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_divehi_bochen0909_pipeline", lang = "dv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_divehi_bochen0909_pipeline", lang = "dv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_divehi_bochen0909_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|dv|
|Size:|1.7 GB|

## References

https://huggingface.co/bochen0909/whisper-small-dv

## Included Models

- AudioAssembler
- WhisperForCTC