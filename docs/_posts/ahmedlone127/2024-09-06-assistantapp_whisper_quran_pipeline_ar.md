---
layout: model
title: Arabic assistantapp_whisper_quran_pipeline pipeline WhisperForCTC from AssistantApp
author: John Snow Labs
name: assistantapp_whisper_quran_pipeline
date: 2024-09-06
tags: [ar, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`assistantapp_whisper_quran_pipeline` is a Arabic model originally trained by AssistantApp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/assistantapp_whisper_quran_pipeline_ar_5.5.0_3.0_1725603926900.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/assistantapp_whisper_quran_pipeline_ar_5.5.0_3.0_1725603926900.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("assistantapp_whisper_quran_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("assistantapp_whisper_quran_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assistantapp_whisper_quran_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|643.1 MB|

## References

https://huggingface.co/AssistantApp/assistantapp-whisper-quran

## Included Models

- AudioAssembler
- WhisperForCTC