---
layout: model
title: Korean clip_vit_large_patch14_korean_pipeline pipeline CLIPForZeroShotClassification from Bingsu
author: John Snow Labs
name: clip_vit_large_patch14_korean_pipeline
date: 2024-09-04
tags: [ko, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_vit_large_patch14_korean_pipeline` is a Korean model originally trained by Bingsu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_vit_large_patch14_korean_pipeline_ko_5.5.0_3.0_1725492221124.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_vit_large_patch14_korean_pipeline_ko_5.5.0_3.0_1725492221124.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clip_vit_large_patch14_korean_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clip_vit_large_patch14_korean_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_vit_large_patch14_korean_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|1.2 GB|

## References

https://huggingface.co/Bingsu/clip-vit-large-patch14-ko

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification