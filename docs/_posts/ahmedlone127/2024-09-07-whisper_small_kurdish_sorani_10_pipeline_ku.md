---
layout: model
title: Kurdish whisper_small_kurdish_sorani_10_pipeline pipeline WhisperForCTC from roshna-omer
author: John Snow Labs
name: whisper_small_kurdish_sorani_10_pipeline
date: 2024-09-07
tags: [ku, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ku
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_kurdish_sorani_10_pipeline` is a Kurdish model originally trained by roshna-omer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_kurdish_sorani_10_pipeline_ku_5.5.0_3.0_1725752882030.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_kurdish_sorani_10_pipeline_ku_5.5.0_3.0_1725752882030.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_kurdish_sorani_10_pipeline", lang = "ku")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_kurdish_sorani_10_pipeline", lang = "ku")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_kurdish_sorani_10_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ku|
|Size:|1.7 GB|

## References

https://huggingface.co/roshna-omer/whisper-small-Kurdish-Sorani-10

## Included Models

- AudioAssembler
- WhisperForCTC