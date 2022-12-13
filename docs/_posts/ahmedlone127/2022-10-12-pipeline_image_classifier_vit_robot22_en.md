---
layout: model
title: English pipeline_image_classifier_vit_robot22 ViTForImageClassification from sudo-s
author: John Snow Labs
name: pipeline_image_classifier_vit_robot22
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_robot22` is a English model originally trained by sudo-s.


## Predicted Entities

`45`, `98`, `113`, `34`, `67`, `120`, `93`, `142`, `147`, `12`, `66`, `89`, `51`, `124`, `84`, `8`, `73`, `78`, `19`, `100`, `23`, `62`, `135`, `128`, `4`, `121`, `88`, `77`, `40`, `110`, `15`, `11`, `104`, `90`, `9`, `141`, `139`, `132`, `44`, `33`, `117`, `22`, `56`, `55`, `26`, `134`, `50`, `123`, `37`, `68`, `61`, `107`, `13`, `46`, `99`, `24`, `94`, `83`, `35`, `16`, `79`, `5`, `103`, `112`, `72`, `10`, `59`, `144`, `87`, `48`, `21`, `116`, `76`, `138`, `54`, `43`, `148`, `127`, `65`, `71`, `57`, `108`, `32`, `80`, `106`, `137`, `82`, `49`, `6`, `126`, `36`, `1`, `39`, `140`, `17`, `25`, `60`, `14`, `133`, `47`, `122`, `111`, `102`, `31`, `96`, `69`, `95`, `58`, `145`, `64`, `53`, `42`, `75`, `115`, `109`, `0`, `20`, `27`, `70`, `2`, `86`, `38`, `81`, `118`, `92`, `125`, `18`, `101`, `30`, `7`, `143`, `97`, `130`, `114`, `129`, `29`, `41`, `105`, `63`, `3`, `74`, `91`, `52`, `85`, `131`, `28`, `119`, `136`, `146`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_robot22_en_4.2.1_3.0_1665536522844.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_robot22_en_4.2.1_3.0_1665536522844.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_robot22', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_robot22", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_robot22|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.3 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification