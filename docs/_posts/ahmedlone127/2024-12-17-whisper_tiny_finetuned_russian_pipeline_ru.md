---
layout: model
title: Russian whisper_tiny_finetuned_russian_pipeline pipeline WhisperForCTC from Ailurus
author: John Snow Labs
name: whisper_tiny_finetuned_russian_pipeline
date: 2024-12-17
tags: [ru, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_finetuned_russian_pipeline` is a Russian model originally trained by Ailurus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_finetuned_russian_pipeline_ru_5.5.1_3.0_1734400638004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_finetuned_russian_pipeline_ru_5.5.1_3.0_1734400638004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_finetuned_russian_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_finetuned_russian_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_finetuned_russian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|389.7 MB|

## References

https://huggingface.co/Ailurus/whisper-tiny-finetuned-ru

## Included Models

- AudioAssembler
- WhisperForCTC