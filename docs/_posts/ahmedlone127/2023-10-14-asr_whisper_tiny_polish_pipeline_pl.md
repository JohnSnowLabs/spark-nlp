---
layout: model
title: Polish asr_whisper_tiny_polish_pipeline pipeline WhisperForCTC from Aspik101
author: John Snow Labs
name: asr_whisper_tiny_polish_pipeline
date: 2023-10-14
tags: [whisper, pl, open_source, pipeline]
task: Automatic Speech Recognition
language: pl
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_tiny_polish_pipeline` is a Polish model originally trained by Aspik101.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_tiny_polish_pipeline_pl_5.1.4_3.4_1697247473830.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_tiny_polish_pipeline_pl_5.1.4_3.4_1697247473830.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_tiny_polish_pipeline', lang = 'pl')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_tiny_polish_pipeline', lang = 'pl')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_tiny_polish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|pl|
|Size:|390.7 MB|

## References

https://huggingface.co/Aspik101/whisper-tiny-pl

## Included Models

- AudioAssembler
- WhisperForCTC