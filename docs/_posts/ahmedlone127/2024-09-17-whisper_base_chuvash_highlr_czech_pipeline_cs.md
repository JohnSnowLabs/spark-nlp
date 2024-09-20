---
layout: model
title: Czech whisper_base_chuvash_highlr_czech_pipeline pipeline WhisperForCTC from sgangireddy
author: John Snow Labs
name: whisper_base_chuvash_highlr_czech_pipeline
date: 2024-09-17
tags: [cs, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: cs
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_base_chuvash_highlr_czech_pipeline` is a Czech model originally trained by sgangireddy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_base_chuvash_highlr_czech_pipeline_cs_5.5.0_3.0_1726549292006.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_base_chuvash_highlr_czech_pipeline_cs_5.5.0_3.0_1726549292006.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_base_chuvash_highlr_czech_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_base_chuvash_highlr_czech_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_base_chuvash_highlr_czech_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|642.6 MB|

## References

https://huggingface.co/sgangireddy/whisper-base-cv-highLR-cs

## Included Models

- AudioAssembler
- WhisperForCTC