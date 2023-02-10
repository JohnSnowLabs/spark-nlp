---
layout: model
title: English pipeline_image_classifier_vit_base_food101 ViTForImageClassification from eslamxm
author: John Snow Labs
name: pipeline_image_classifier_vit_base_food101
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_base_food101` is a English model originally trained by eslamxm.


## Predicted Entities

`grilled_cheese_sandwich`, `edamame`, `onion_rings`, `french_onion_soup`, `french_fries`, `creme_brulee`, `lobster_roll_sandwich`, `bruschetta`, `breakfast_burrito`, `caprese_salad`, `churros`, `omelette`, `club_sandwich`, `chocolate_mousse`, `nachos`, `bread_pudding`, `steak`, `hummus`, `panna_cotta`, `filet_mignon`, `sashimi`, `hot_and_sour_soup`, `cannoli`, `ravioli`, `samosa`, `grilled_salmon`, `lobster_bisque`, `seaweed_salad`, `macaroni_and_cheese`, `fish_and_chips`, `caesar_salad`, `dumplings`, `baby_back_ribs`, `fried_rice`, `oysters`, `peking_duck`, `guacamole`, `greek_salad`, `donuts`, `risotto`, `escargots`, `crab_cakes`, `waffles`, `carrot_cake`, `prime_rib`, `tuna_tartare`, `pho`, `chocolate_cake`, `bibimbap`, `fried_calamari`, `spaghetti_bolognese`, `gnocchi`, `chicken_quesadilla`, `frozen_yogurt`, `apple_pie`, `baklava`, `pulled_pork_sandwich`, `clam_chowder`, `eggs_benedict`, `lasagna`, `ceviche`, `paella`, `foie_gras`, `spring_rolls`, `falafel`, `miso_soup`, `pork_chop`, `ramen`, `pad_thai`, `garlic_bread`, `macarons`, `ice_cream`, `mussels`, `chicken_wings`, `pancakes`, `gyoza`, `poutine`, `croque_madame`, `pizza`, `cheese_plate`, `beignets`, `huevos_rancheros`, `french_toast`, `sushi`, `takoyaki`, `spaghetti_carbonara`, `beef_tartare`, `scallops`, `cup_cakes`, `tacos`, `deviled_eggs`, `beet_salad`, `tiramisu`, `cheesecake`, `strawberry_shortcake`, `beef_carpaccio`, `hamburger`, `red_velvet_cake`, `hot_dog`, `shrimp_and_grits`, `chicken_curry`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_food101_en_4.2.1_3.0_1665537813980.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_food101_en_4.2.1_3.0_1665537813980.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_base_food101', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_base_food101", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_base_food101|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.2 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification