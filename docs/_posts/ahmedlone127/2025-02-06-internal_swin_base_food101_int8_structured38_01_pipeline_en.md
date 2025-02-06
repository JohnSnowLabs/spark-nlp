---
layout: model
title: English internal_swin_base_food101_int8_structured38_01_pipeline pipeline SwinForImageClassification from yujiepan
author: John Snow Labs
name: internal_swin_base_food101_int8_structured38_01_pipeline
date: 2025-02-06
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`internal_swin_base_food101_int8_structured38_01_pipeline` is a English model originally trained by yujiepan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/internal_swin_base_food101_int8_structured38_01_pipeline_en_5.5.1_3.0_1738843647802.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/internal_swin_base_food101_int8_structured38_01_pipeline_en_5.5.1_3.0_1738843647802.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("internal_swin_base_food101_int8_structured38_01_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("internal_swin_base_food101_int8_structured38_01_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|internal_swin_base_food101_int8_structured38_01_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|647.8 MB|

## References

https://huggingface.co/yujiepan/internal.swin-base-food101-int8-structured38.01

## Included Models

- ImageAssembler
- SwinForImageClassification