---
layout: model
title: English food_type_image_detection_nepal_bhasa_dima806_pipeline pipeline ViTForImageClassification from dima806
author: John Snow Labs
name: food_type_image_detection_nepal_bhasa_dima806_pipeline
date: 2025-02-04
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`food_type_image_detection_nepal_bhasa_dima806_pipeline` is a English model originally trained by dima806.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/food_type_image_detection_nepal_bhasa_dima806_pipeline_en_5.5.1_3.0_1738681629341.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/food_type_image_detection_nepal_bhasa_dima806_pipeline_en_5.5.1_3.0_1738681629341.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("food_type_image_detection_nepal_bhasa_dima806_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("food_type_image_detection_nepal_bhasa_dima806_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|food_type_image_detection_nepal_bhasa_dima806_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.4 MB|

## References

https://huggingface.co/dima806/food_type_image_detection_new

## Included Models

- ImageAssembler
- ViTForImageClassification