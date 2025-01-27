---
layout: model
title: English fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline pipeline ViTForImageClassification from touchtech
author: John Snow Labs
name: fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline` is a English model originally trained by touchtech.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline_en_5.5.1_3.0_1737974591959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline_en_5.5.1_3.0_1737974591959.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fashion_images_gender_age_vit_large_patch16_224_in21k_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/touchtech/fashion-images-gender-age-vit-large-patch16-224-in21k-v2

## Included Models

- ImageAssembler
- ViTForImageClassification