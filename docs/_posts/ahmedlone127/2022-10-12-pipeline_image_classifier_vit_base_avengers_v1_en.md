---
layout: model
title: English pipeline_image_classifier_vit_base_avengers_v1 ViTForImageClassification from dingusagar
author: John Snow Labs
name: pipeline_image_classifier_vit_base_avengers_v1
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
nav_key: models
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_base_avengers_v1` is a English model originally trained by dingusagar.


## Predicted Entities

`Thor`, `Captain America`, `Captain Marvel`, `Falcon Avengers`, `Vision Avengers`, `Bucky Barnes`, `Loki`, `Black Widow`, `Iron Man`, `Black Panther`, `Docter Strage`, `Scarlet Witch`, `Ant Man`, `Hulk`, `Spider Man`, `Hawkeye Avengers`



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_avengers_v1_en_4.2.1_3.0_1665569662969.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_avengers_v1_en_4.2.1_3.0_1665569662969.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_base_avengers_v1', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_base_avengers_v1", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_base_avengers_v1|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.0 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification
