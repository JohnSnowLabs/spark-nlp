---
layout: model
title: English whisper_medium_uzbek_extra_dataset_v2_pipeline pipeline WhisperForCTC from blackhole33
author: John Snow Labs
name: whisper_medium_uzbek_extra_dataset_v2_pipeline
date: 2024-09-14
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_uzbek_extra_dataset_v2_pipeline` is a English model originally trained by blackhole33.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_uzbek_extra_dataset_v2_pipeline_en_5.5.0_3.0_1726333665763.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_uzbek_extra_dataset_v2_pipeline_en_5.5.0_3.0_1726333665763.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_uzbek_extra_dataset_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_uzbek_extra_dataset_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_uzbek_extra_dataset_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|4.8 GB|

## References

https://huggingface.co/blackhole33/whisper-medium-uz-extra_dataset.v2

## Included Models

- AudioAssembler
- WhisperForCTC