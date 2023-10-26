---
layout: model
title: Vietnamese asr_whisper_small_vietnamese_tuananh7198_pipeline pipeline WhisperForCTC from tuananh7198
author: John Snow Labs
name: asr_whisper_small_vietnamese_tuananh7198_pipeline
date: 2023-10-18
tags: [whisper, vi, open_source, pipeline]
task: Automatic Speech Recognition
language: vi
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_vietnamese_tuananh7198_pipeline` is a Vietnamese model originally trained by tuananh7198.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_vietnamese_tuananh7198_pipeline_vi_5.1.4_3.4_1697623445458.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_vietnamese_tuananh7198_pipeline_vi_5.1.4_3.4_1697623445458.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_small_vietnamese_tuananh7198_pipeline', lang = 'vi')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_small_vietnamese_tuananh7198_pipeline', lang = 'vi')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_vietnamese_tuananh7198_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|1.7 GB|

## References

https://huggingface.co/tuananh7198/whisper-small-vi

## Included Models

- AudioAssembler
- WhisperForCTC