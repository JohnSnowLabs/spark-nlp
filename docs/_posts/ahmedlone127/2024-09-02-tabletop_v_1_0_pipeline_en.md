---
layout: model
title: English tabletop_v_1_0_pipeline pipeline CLIPForZeroShotClassification from orionai
author: John Snow Labs
name: tabletop_v_1_0_pipeline
date: 2024-09-02
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tabletop_v_1_0_pipeline` is a English model originally trained by orionai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tabletop_v_1_0_pipeline_en_5.5.0_3.0_1725276357870.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tabletop_v_1_0_pipeline_en_5.5.0_3.0_1725276357870.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tabletop_v_1_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tabletop_v_1_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tabletop_v_1_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|567.3 MB|

## References

https://huggingface.co/orionai/tabletop-v.1.0

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification