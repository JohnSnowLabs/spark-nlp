---
layout: model
title: English vit_l_tiny_imagenet_pipeline pipeline ViTForImageClassification from hiendang7613
author: John Snow Labs
name: vit_l_tiny_imagenet_pipeline
date: 2025-01-30
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_l_tiny_imagenet_pipeline` is a English model originally trained by hiendang7613.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_l_tiny_imagenet_pipeline_en_5.5.1_3.0_1738232549273.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_l_tiny_imagenet_pipeline_en_5.5.1_3.0_1738232549273.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_l_tiny_imagenet_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_l_tiny_imagenet_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_l_tiny_imagenet_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/hiendang7613/vit-l-tiny-imagenet

## Included Models

- ImageAssembler
- ViTForImageClassification