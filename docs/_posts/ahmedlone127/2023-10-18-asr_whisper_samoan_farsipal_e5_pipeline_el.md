---
layout: model
title: Modern Greek (1453-) asr_whisper_samoan_farsipal_e5_pipeline pipeline WhisperForCTC from emilios
author: John Snow Labs
name: asr_whisper_samoan_farsipal_e5_pipeline
date: 2023-10-18
tags: [whisper, el, open_source, pipeline]
task: Automatic Speech Recognition
language: el
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_samoan_farsipal_e5_pipeline` is a Modern Greek (1453-) model originally trained by emilios.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_samoan_farsipal_e5_pipeline_el_5.1.4_3.4_1697589687809.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_samoan_farsipal_e5_pipeline_el_5.1.4_3.4_1697589687809.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_samoan_farsipal_e5_pipeline', lang = 'el')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_samoan_farsipal_e5_pipeline', lang = 'el')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_samoan_farsipal_e5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|1.1 GB|

## References

https://huggingface.co/emilios/whisper-sm-farsipal-e5

## Included Models

- AudioAssembler
- WhisperForCTC