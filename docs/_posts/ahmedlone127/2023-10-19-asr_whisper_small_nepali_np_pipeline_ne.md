---
layout: model
title: Nepali (macrolanguage) asr_whisper_small_nepali_np_pipeline pipeline WhisperForCTC from julie200
author: John Snow Labs
name: asr_whisper_small_nepali_np_pipeline
date: 2023-10-19
tags: [whisper, ne, open_source, pipeline]
task: Automatic Speech Recognition
language: ne
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_nepali_np_pipeline` is a Nepali (macrolanguage) model originally trained by julie200.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_nepali_np_pipeline_ne_5.1.4_3.4_1697758262511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_nepali_np_pipeline_ne_5.1.4_3.4_1697758262511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_small_nepali_np_pipeline', lang = 'ne')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_small_nepali_np_pipeline', lang = 'ne')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_nepali_np_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|ne|
|Size:|1.7 GB|

## References

https://huggingface.co/julie200/whisper-small-ne-np

## Included Models

- AudioAssembler
- WhisperForCTC