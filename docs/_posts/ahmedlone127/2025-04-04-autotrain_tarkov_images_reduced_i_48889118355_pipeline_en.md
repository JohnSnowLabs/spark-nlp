---
layout: model
title: English autotrain_tarkov_images_reduced_i_48889118355_pipeline pipeline ViTForImageClassification from friesel
author: John Snow Labs
name: autotrain_tarkov_images_reduced_i_48889118355_pipeline
date: 2025-04-04
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_tarkov_images_reduced_i_48889118355_pipeline` is a English model originally trained by friesel.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_tarkov_images_reduced_i_48889118355_pipeline_en_5.5.1_3.0_1743748237361.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_tarkov_images_reduced_i_48889118355_pipeline_en_5.5.1_3.0_1743748237361.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_tarkov_images_reduced_i_48889118355_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_tarkov_images_reduced_i_48889118355_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_tarkov_images_reduced_i_48889118355_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.4 MB|

## References

https://huggingface.co/friesel/autotrain-tarkov_images_reduced_i-48889118355

## Included Models

- ImageAssembler
- ViTForImageClassification