---
layout: model
title: English pipeline_image_classifier_vit_mit_indoor_scenes ViTForImageClassification from vincentclaes
author: John Snow Labs
name: pipeline_image_classifier_vit_mit_indoor_scenes
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_mit_indoor_scenes` is a English model originally trained by vincentclaes.


## Predicted Entities

`airport_inside`, `bowling`, `buffet`, `movietheater`, `clothingstore`, `inside_bus`, `fastfood_restaurant`, `operating_room`, `corridor`, `cloister`, `stairscase`, `auditorium`, `meeting_room`, `livingroom`, `videostore`, `bathroom`, `inside_subway`, `bedroom`, `casino`, `tv_studio`, `classroom`, `laboratorywet`, `nursery`, `office`, `deli`, `prisoncell`, `dentaloffice`, `restaurant_kitchen`, `studiomusic`, `locker_room`, `restaurant`, `laundromat`, `dining_room`, `subway`, `gameroom`, `museum`, `mall`, `garage`, `elevator`, `jewelleryshop`, `kindergarden`, `toystore`, `concert_hall`, `artstudio`, `kitchen`, `florist`, `waitingroom`, `grocerystore`, `library`, `bar`, `computerroom`, `trainstation`, `lobby`, `church_inside`, `pantry`, `closet`, `children_room`, `hairsalon`, `shoeshop`, `greenhouse`, `bookstore`, `bakery`, `poolinside`, `warehouse`, `winecellar`, `hospitalroom`, `gym`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_mit_indoor_scenes_en_4.2.1_3.0_1665569637323.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_mit_indoor_scenes', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_mit_indoor_scenes", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_mit_indoor_scenes|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.1 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification