---
layout: model
title: Arabic graduation_project_whisper_base_pipeline pipeline WhisperForCTC from YoussefAshmawy
author: John Snow Labs
name: graduation_project_whisper_base_pipeline
date: 2024-12-17
tags: [ar, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`graduation_project_whisper_base_pipeline` is a Arabic model originally trained by YoussefAshmawy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/graduation_project_whisper_base_pipeline_ar_5.5.1_3.0_1734401629610.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/graduation_project_whisper_base_pipeline_ar_5.5.1_3.0_1734401629610.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("graduation_project_whisper_base_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("graduation_project_whisper_base_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|graduation_project_whisper_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|640.4 MB|

## References

https://huggingface.co/YoussefAshmawy/Graduation_Project_Whisper_base

## Included Models

- AudioAssembler
- WhisperForCTC