---
layout: model
title: English flip_base_16_pipeline pipeline CLIPForZeroShotClassification from FLIP-dataset
author: John Snow Labs
name: flip_base_16_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`flip_base_16_pipeline` is a English model originally trained by FLIP-dataset.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/flip_base_16_pipeline_en_5.5.0_3.0_1725338281908.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/flip_base_16_pipeline_en_5.5.0_3.0_1725338281908.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("flip_base_16_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("flip_base_16_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|flip_base_16_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|561.2 MB|

## References

https://huggingface.co/FLIP-dataset/FLIP-base-16

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification