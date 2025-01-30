---
layout: model
title: English google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline pipeline ViTForImageClassification from amaye15
author: John Snow Labs
name: google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline
date: 2025-01-27
tags: [en, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline` is a English model originally trained by amaye15.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline_en_5.5.1_3.0_1737975174503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline_en_5.5.1_3.0_1737975174503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_vit_base_patch16_224_batch32_lr5e_05_standford_dogs_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.7 MB|

## References

https://huggingface.co/amaye15/google-vit-base-patch16-224-batch32-lr5e-05-standford-dogs

## Included Models

- ImageAssembler
- ViTForImageClassification