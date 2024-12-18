---
layout: model
title: Finnish whisper_small_finnish_vapari_pipeline pipeline WhisperForCTC from vapari
author: John Snow Labs
name: whisper_small_finnish_vapari_pipeline
date: 2024-12-17
tags: [fi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_finnish_vapari_pipeline` is a Finnish model originally trained by vapari.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_finnish_vapari_pipeline_fi_5.5.1_3.0_1734400647993.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_finnish_vapari_pipeline_fi_5.5.1_3.0_1734400647993.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_finnish_vapari_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_finnish_vapari_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_finnish_vapari_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|1.7 GB|

## References

https://huggingface.co/vapari/whisper-small-fi

## Included Models

- AudioAssembler
- WhisperForCTC