---
layout: model
title: Basque whisper_tiny_basque_cv16_1_pipeline pipeline WhisperForCTC from zuazo
author: John Snow Labs
name: whisper_tiny_basque_cv16_1_pipeline
date: 2024-09-17
tags: [eu, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: eu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_basque_cv16_1_pipeline` is a Basque model originally trained by zuazo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_basque_cv16_1_pipeline_eu_5.5.0_3.0_1726558379352.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_basque_cv16_1_pipeline_eu_5.5.0_3.0_1726558379352.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_basque_cv16_1_pipeline", lang = "eu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_basque_cv16_1_pipeline", lang = "eu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_basque_cv16_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|eu|
|Size:|389.4 MB|

## References

https://huggingface.co/zuazo/whisper-tiny-eu-cv16_1

## Included Models

- AudioAssembler
- WhisperForCTC