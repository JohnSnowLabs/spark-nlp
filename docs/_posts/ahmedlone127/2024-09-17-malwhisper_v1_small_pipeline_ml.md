---
layout: model
title: Malayalam malwhisper_v1_small_pipeline pipeline WhisperForCTC from smcproject
author: John Snow Labs
name: malwhisper_v1_small_pipeline
date: 2024-09-17
tags: [ml, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ml
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malwhisper_v1_small_pipeline` is a Malayalam model originally trained by smcproject.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malwhisper_v1_small_pipeline_ml_5.5.0_3.0_1726542432862.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malwhisper_v1_small_pipeline_ml_5.5.0_3.0_1726542432862.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malwhisper_v1_small_pipeline", lang = "ml")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malwhisper_v1_small_pipeline", lang = "ml")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malwhisper_v1_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ml|
|Size:|1.7 GB|

## References

https://huggingface.co/smcproject/Malwhisper-v1-small

## Included Models

- AudioAssembler
- WhisperForCTC