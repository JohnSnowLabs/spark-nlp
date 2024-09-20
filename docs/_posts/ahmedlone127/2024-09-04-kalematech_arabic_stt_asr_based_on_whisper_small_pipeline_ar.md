---
layout: model
title: Arabic kalematech_arabic_stt_asr_based_on_whisper_small_pipeline pipeline WhisperForCTC from Salama1429
author: John Snow Labs
name: kalematech_arabic_stt_asr_based_on_whisper_small_pipeline
date: 2024-09-04
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kalematech_arabic_stt_asr_based_on_whisper_small_pipeline` is a Arabic model originally trained by Salama1429.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kalematech_arabic_stt_asr_based_on_whisper_small_pipeline_ar_5.5.0_3.0_1725429350966.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kalematech_arabic_stt_asr_based_on_whisper_small_pipeline_ar_5.5.0_3.0_1725429350966.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kalematech_arabic_stt_asr_based_on_whisper_small_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kalematech_arabic_stt_asr_based_on_whisper_small_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kalematech_arabic_stt_asr_based_on_whisper_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|1.7 GB|

## References

https://huggingface.co/Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small

## Included Models

- AudioAssembler
- WhisperForCTC