---
layout: model
title: English asr_whisper_small_chinese_hk_pipeline pipeline WhisperForCTC from tilos
author: John Snow Labs
name: asr_whisper_small_chinese_hk_pipeline
date: 2023-10-18
tags: [whisper, en, open_source, pipeline]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_chinese_hk_pipeline` is a English model originally trained by tilos.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_chinese_hk_pipeline_en_5.1.4_3.4_1697588819567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_chinese_hk_pipeline_en_5.1.4_3.4_1697588819567.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_small_chinese_hk_pipeline', lang = 'en')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_small_chinese_hk_pipeline', lang = 'en')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_chinese_hk_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## References

https://huggingface.co/tilos/whisper-small-zh-HK

## Included Models

- AudioAssembler
- WhisperForCTC