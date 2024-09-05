---
layout: model
title: English banglaasr_bangla_speech_processing_pipeline pipeline WhisperForCTC from bangla-speech-processing
author: John Snow Labs
name: banglaasr_bangla_speech_processing_pipeline
date: 2024-09-05
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`banglaasr_bangla_speech_processing_pipeline` is a English model originally trained by bangla-speech-processing.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/banglaasr_bangla_speech_processing_pipeline_en_5.5.0_3.0_1725548179765.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/banglaasr_bangla_speech_processing_pipeline_en_5.5.0_3.0_1725548179765.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("banglaasr_bangla_speech_processing_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("banglaasr_bangla_speech_processing_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|banglaasr_bangla_speech_processing_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## References

https://huggingface.co/bangla-speech-processing/BanglaASR

## Included Models

- AudioAssembler
- WhisperForCTC