---
layout: model
title: English pipeline_image_classifier_vit_gtsrb_model ViTForImageClassification from bazyl
author: John Snow Labs
name: pipeline_image_classifier_vit_gtsrb_model
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_gtsrb_model` is a English model originally trained by bazyl.


## Predicted Entities

`Children crossing`, `Double curve`, `Road work`, `Yield`, `Beware of ice/snow`, `Speed limit (70km/h)`, `Bicycles crossing`, `Roundabout mandatory`, `Speed limit (30km/h)`, `Keep left`, `Dangerous curve left`, `No vehicles`, `End of no passing`, `Bumpy road`, `Speed limit (50km/h)`, `Turn left ahead`, `Speed limit (20km/h)`, `General caution`, `Speed limit (100km/h)`, `End speed + passing limits`, `Go straight or right`, `Dangerous curve right`, `Speed limit (80km/h)`, `Slippery road`, `Turn right ahead`, `No passing veh over 3.5 tons`, `Speed limit (60km/h)`, `Pedestrians`, `Right-of-way at intersection`, `Priority road`, `End of speed limit (80km/h)`, `Road narrows on the right`, `No entry`, `Stop`, `Wild animals crossing`, `Veh > 3.5 tons prohibited`, `End no passing veh > 3.5 tons`, `Go straight or left`, `Speed limit (120km/h)`, `Ahead only`, `Keep right`, `Traffic signals`, `No passing`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_gtsrb_model_en_4.2.1_3.0_1665570210978.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_gtsrb_model_en_4.2.1_3.0_1665570210978.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_gtsrb_model', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_gtsrb_model", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_gtsrb_model|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.0 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification