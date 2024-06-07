---
layout: model
title: English clip_vit_large_patch14_pipeline pipeline CLIPForZeroShotClassification from openai
author: John Snow Labs
name: clip_vit_large_patch14_pipeline
date: 2024-06-07
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_vit_large_patch14_pipeline` is a English model originally trained by openai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_vit_large_patch14_pipeline_en_5.2.4_3.0_1717784838588.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_vit_large_patch14_pipeline_en_5.2.4_3.0_1717784838588.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('clip_vit_large_patch14_pipeline', lang = 'en')
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline('clip_vit_large_patch14_pipeline', lang = 'en')
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_vit_large_patch14_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/openai/clip-vit-large-patch14

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification