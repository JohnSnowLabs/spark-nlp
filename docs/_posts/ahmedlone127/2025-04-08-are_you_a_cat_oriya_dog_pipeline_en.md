---
layout: model
title: English are_you_a_cat_oriya_dog_pipeline pipeline ViTForImageClassification from Kang-Seong-Jun
author: John Snow Labs
name: are_you_a_cat_oriya_dog_pipeline
date: 2025-04-08
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`are_you_a_cat_oriya_dog_pipeline` is a English model originally trained by Kang-Seong-Jun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/are_you_a_cat_oriya_dog_pipeline_en_5.5.1_3.0_1744097981100.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/are_you_a_cat_oriya_dog_pipeline_en_5.5.1_3.0_1744097981100.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("are_you_a_cat_oriya_dog_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("are_you_a_cat_oriya_dog_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|are_you_a_cat_oriya_dog_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/Kang-Seong-Jun/ARE-YOU-A-CAT-OR-DOG

## Included Models

- ImageAssembler
- ViTForImageClassification