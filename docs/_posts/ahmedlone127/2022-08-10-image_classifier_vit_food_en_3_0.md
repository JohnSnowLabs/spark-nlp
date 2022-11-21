---
layout: model
title: English image_classifier_vit_food ViTForImageClassification from nateraw
author: John Snow Labs
name: image_classifier_vit_food
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_food` is a English model originally trained by nateraw.


## Predicted Entities

`grilled_cheese_sandwich`, `edamame`, `onion_rings`, `french_onion_soup`, `french_fries`, `creme_brulee`, `lobster_roll_sandwich`, `bruschetta`, `breakfast_burrito`, `caprese_salad`, `churros`, `omelette`, `club_sandwich`, `chocolate_mousse`, `nachos`, `bread_pudding`, `steak`, `hummus`, `panna_cotta`, `filet_mignon`, `sashimi`, `hot_and_sour_soup`, `cannoli`, `ravioli`, `samosa`, `grilled_salmon`, `lobster_bisque`, `seaweed_salad`, `macaroni_and_cheese`, `fish_and_chips`, `caesar_salad`, `dumplings`, `baby_back_ribs`, `fried_rice`, `oysters`, `peking_duck`, `guacamole`, `greek_salad`, `donuts`, `risotto`, `escargots`, `crab_cakes`, `waffles`, `carrot_cake`, `prime_rib`, `tuna_tartare`, `pho`, `chocolate_cake`, `bibimbap`, `fried_calamari`, `spaghetti_bolognese`, `gnocchi`, `chicken_quesadilla`, `frozen_yogurt`, `apple_pie`, `baklava`, `pulled_pork_sandwich`, `clam_chowder`, `eggs_benedict`, `lasagna`, `ceviche`, `paella`, `foie_gras`, `spring_rolls`, `falafel`, `miso_soup`, `pork_chop`, `ramen`, `pad_thai`, `garlic_bread`, `macarons`, `ice_cream`, `mussels`, `chicken_wings`, `pancakes`, `gyoza`, `poutine`, `croque_madame`, `pizza`, `cheese_plate`, `beignets`, `huevos_rancheros`, `french_toast`, `sushi`, `takoyaki`, `spaghetti_carbonara`, `beef_tartare`, `scallops`, `cup_cakes`, `tacos`, `deviled_eggs`, `beet_salad`, `tiramisu`, `cheesecake`, `strawberry_shortcake`, `beef_carpaccio`, `hamburger`, `red_velvet_cake`, `hot_dog`, `shrimp_and_grits`, `chicken_curry`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_food_en_4.1.0_3.0_1660167590552.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_food", "en")\
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
.pretrained("image_classifier_vit_food", "en")\
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
|Model Name:|image_classifier_vit_food|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.2 MB|