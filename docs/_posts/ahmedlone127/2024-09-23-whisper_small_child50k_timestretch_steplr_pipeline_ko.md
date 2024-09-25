---
layout: model
title: Korean whisper_small_child50k_timestretch_steplr_pipeline pipeline WhisperForCTC from haseong8012
author: John Snow Labs
name: whisper_small_child50k_timestretch_steplr_pipeline
date: 2024-09-23
tags: [ko, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_child50k_timestretch_steplr_pipeline` is a Korean model originally trained by haseong8012.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_child50k_timestretch_steplr_pipeline_ko_5.5.0_3.0_1727052228583.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_child50k_timestretch_steplr_pipeline_ko_5.5.0_3.0_1727052228583.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_child50k_timestretch_steplr_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_child50k_timestretch_steplr_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_child50k_timestretch_steplr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|1.7 GB|

## References

https://huggingface.co/haseong8012/whisper-small_child50K_timestretch_stepLR

## Included Models

- AudioAssembler
- WhisperForCTC