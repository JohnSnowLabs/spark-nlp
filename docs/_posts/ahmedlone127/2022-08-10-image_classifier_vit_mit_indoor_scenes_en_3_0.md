---
layout: model
title: English image_classifier_vit_mit_indoor_scenes ViTForImageClassification from vincentclaes
author: John Snow Labs
name: image_classifier_vit_mit_indoor_scenes
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: ViTForImageClassification
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_mit_indoor_scenes_en_4.1.0_3.0_1660170571863.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_vit_mit_indoor_scenes_en_4.1.0_3.0_1660170571863.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_mit_indoor_scenes", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_mit_indoor_scenes", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_mit_indoor_scenes|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.1 MB|