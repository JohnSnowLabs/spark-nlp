---
layout: model
title: Persian vit_persian_food_classifier_mini_pipeline pipeline ViTForImageClassification from iTzMiNOS
author: John Snow Labs
name: vit_persian_food_classifier_mini_pipeline
date: 2025-04-08
tags: [fa, open_source, pipeline, onnx]
task: Image Classification
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_persian_food_classifier_mini_pipeline` is a Persian model originally trained by iTzMiNOS.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_persian_food_classifier_mini_pipeline_fa_5.5.1_3.0_1744098661209.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_persian_food_classifier_mini_pipeline_fa_5.5.1_3.0_1744098661209.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_persian_food_classifier_mini_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_persian_food_classifier_mini_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_persian_food_classifier_mini_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|321.4 MB|

## References

https://huggingface.co/iTzMiNOS/vit-persian-food-classifier-mini

## Included Models

- ImageAssembler
- ViTForImageClassification