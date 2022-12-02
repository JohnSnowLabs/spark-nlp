---
layout: model
title: English image_classifier_vit_gtsrb_model ViTForImageClassification from bazyl
author: John Snow Labs
name: image_classifier_vit_gtsrb_model
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_gtsrb_model` is a English model originally trained by bazyl.


## Predicted Entities

`Children crossing`, `Double curve`, `Road work`, `Yield`, `Beware of ice/snow`, `Speed limit (70km/h)`, `Bicycles crossing`, `Roundabout mandatory`, `Speed limit (30km/h)`, `Keep left`, `Dangerous curve left`, `No vehicles`, `End of no passing`, `Bumpy road`, `Speed limit (50km/h)`, `Turn left ahead`, `Speed limit (20km/h)`, `General caution`, `Speed limit (100km/h)`, `End speed + passing limits`, `Go straight or right`, `Dangerous curve right`, `Speed limit (80km/h)`, `Slippery road`, `Turn right ahead`, `No passing veh over 3.5 tons`, `Speed limit (60km/h)`, `Pedestrians`, `Right-of-way at intersection`, `Priority road`, `End of speed limit (80km/h)`, `Road narrows on the right`, `No entry`, `Stop`, `Wild animals crossing`, `Veh > 3.5 tons prohibited`, `End no passing veh > 3.5 tons`, `Go straight or left`, `Speed limit (120km/h)`, `Ahead only`, `Keep right`, `Traffic signals`, `No passing`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_gtsrb_model_en_4.1.0_3.0_1660166650134.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_gtsrb_model", "en")\
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
.pretrained("image_classifier_vit_gtsrb_model", "en")\
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
|Model Name:|image_classifier_vit_gtsrb_model|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.0 MB|