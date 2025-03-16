---
layout: model
title: English clip_vit_b_32_laion2b_s34b_b79k_pipeline pipeline CLIPForZeroShotClassification from timsek
author: John Snow Labs
name: clip_vit_b_32_laion2b_s34b_b79k_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_vit_b_32_laion2b_s34b_b79k_pipeline` is a English model originally trained by timsek.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_vit_b_32_laion2b_s34b_b79k_pipeline_en_5.5.1_3.0_1737629447482.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_vit_b_32_laion2b_s34b_b79k_pipeline_en_5.5.1_3.0_1737629447482.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clip_vit_b_32_laion2b_s34b_b79k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clip_vit_b_32_laion2b_s34b_b79k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_vit_b_32_laion2b_s34b_b79k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|567.6 MB|

## References

https://huggingface.co/timsek/CLIP-ViT-B-32-laion2B-s34B-b79K

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification