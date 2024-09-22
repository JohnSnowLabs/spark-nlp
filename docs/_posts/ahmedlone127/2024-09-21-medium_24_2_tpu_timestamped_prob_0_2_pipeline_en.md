---
layout: model
title: English medium_24_2_tpu_timestamped_prob_0_2_pipeline pipeline WhisperForCTC from sanchit-gandhi
author: John Snow Labs
name: medium_24_2_tpu_timestamped_prob_0_2_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`medium_24_2_tpu_timestamped_prob_0_2_pipeline` is a English model originally trained by sanchit-gandhi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/medium_24_2_tpu_timestamped_prob_0_2_pipeline_en_5.5.0_3.0_1726912899567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/medium_24_2_tpu_timestamped_prob_0_2_pipeline_en_5.5.0_3.0_1726912899567.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("medium_24_2_tpu_timestamped_prob_0_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("medium_24_2_tpu_timestamped_prob_0_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medium_24_2_tpu_timestamped_prob_0_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.8 GB|

## References

https://huggingface.co/sanchit-gandhi/medium-24-2-tpu-timestamped-prob-0.2

## Included Models

- AudioAssembler
- WhisperForCTC