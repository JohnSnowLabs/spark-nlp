---
layout: model
title: Lithuanian common_voice_pipeline pipeline WhisperForCTC from Tomas1234
author: John Snow Labs
name: common_voice_pipeline
date: 2024-09-19
tags: [lt, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: lt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`common_voice_pipeline` is a Lithuanian model originally trained by Tomas1234.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/common_voice_pipeline_lt_5.5.0_3.0_1726757318113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/common_voice_pipeline_lt_5.5.0_3.0_1726757318113.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("common_voice_pipeline", lang = "lt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("common_voice_pipeline", lang = "lt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|common_voice_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|lt|
|Size:|1.7 GB|

## References

https://huggingface.co/Tomas1234/common_voice

## Included Models

- AudioAssembler
- WhisperForCTC