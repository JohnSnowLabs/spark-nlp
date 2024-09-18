---
layout: model
title: Arabic whisper_medium_arabic_abosteet_pipeline pipeline WhisperForCTC from Abosteet
author: John Snow Labs
name: whisper_medium_arabic_abosteet_pipeline
date: 2024-09-13
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_arabic_abosteet_pipeline` is a Arabic model originally trained by Abosteet.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_arabic_abosteet_pipeline_ar_5.5.0_3.0_1726223380335.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_arabic_abosteet_pipeline_ar_5.5.0_3.0_1726223380335.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_arabic_abosteet_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_arabic_abosteet_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_arabic_abosteet_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|4.8 GB|

## References

https://huggingface.co/Abosteet/whisper-medium-arabic

## Included Models

- AudioAssembler
- WhisperForCTC