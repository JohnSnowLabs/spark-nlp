---
layout: model
title: English distill_whisper_thai_medium_bjak_pipeline pipeline WhisperForCTC from bjak
author: John Snow Labs
name: distill_whisper_thai_medium_bjak_pipeline
date: 2024-09-09
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distill_whisper_thai_medium_bjak_pipeline` is a English model originally trained by bjak.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distill_whisper_thai_medium_bjak_pipeline_en_5.5.0_3.0_1725843975427.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distill_whisper_thai_medium_bjak_pipeline_en_5.5.0_3.0_1725843975427.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distill_whisper_thai_medium_bjak_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distill_whisper_thai_medium_bjak_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distill_whisper_thai_medium_bjak_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.0 GB|

## References

https://huggingface.co/bjak/distill-whisper-th-medium

## Included Models

- AudioAssembler
- WhisperForCTC