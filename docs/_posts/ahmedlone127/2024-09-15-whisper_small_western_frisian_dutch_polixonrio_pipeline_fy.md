---
layout: model
title: Western Frisian whisper_small_western_frisian_dutch_polixonrio_pipeline pipeline WhisperForCTC from polixonrio
author: John Snow Labs
name: whisper_small_western_frisian_dutch_polixonrio_pipeline
date: 2024-09-15
tags: [fy, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fy
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_western_frisian_dutch_polixonrio_pipeline` is a Western Frisian model originally trained by polixonrio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_western_frisian_dutch_polixonrio_pipeline_fy_5.5.0_3.0_1726411123187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_western_frisian_dutch_polixonrio_pipeline_fy_5.5.0_3.0_1726411123187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_western_frisian_dutch_polixonrio_pipeline", lang = "fy")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_western_frisian_dutch_polixonrio_pipeline", lang = "fy")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_western_frisian_dutch_polixonrio_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fy|
|Size:|1.7 GB|

## References

https://huggingface.co/polixonrio/whisper-small-fy-NL

## Included Models

- AudioAssembler
- WhisperForCTC