---
layout: model
title: Korean asr_whisper_small_korean_fl_pipeline pipeline WhisperForCTC from p4b
author: John Snow Labs
name: asr_whisper_small_korean_fl_pipeline
date: 2023-10-20
tags: [whisper, ko, open_source, pipeline]
task: Automatic Speech Recognition
language: ko
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_korean_fl_pipeline` is a Korean model originally trained by p4b.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_korean_fl_pipeline_ko_5.1.4_3.4_1697768403117.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_korean_fl_pipeline_ko_5.1.4_3.4_1697768403117.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_small_korean_fl_pipeline', lang = 'ko')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_small_korean_fl_pipeline', lang = 'ko')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_korean_fl_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|1.1 GB|

## References

https://huggingface.co/p4b/whisper-small-ko-fl

## Included Models

- AudioAssembler
- WhisperForCTC