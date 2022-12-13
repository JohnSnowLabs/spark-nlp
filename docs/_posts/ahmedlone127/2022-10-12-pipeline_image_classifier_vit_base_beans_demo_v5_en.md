---
layout: model
title: English pipeline_image_classifier_vit_base_beans_demo_v5 ViTForImageClassification from mrgiraffe
author: John Snow Labs
name: pipeline_image_classifier_vit_base_beans_demo_v5
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_base_beans_demo_v5` is a English model originally trained by mrgiraffe.


## Predicted Entities

`lion`, `tulip`, `keyboard`, `cra`, `bus`, `dolphin`, `plate`, `beaver`, `skyscraper`, `tiger`, `bear`, `trout`, `porcupine`, `sea`, `shrew`, `squirrel`, `snail`, `leopard`, `palm_tree`, `turtle`, `orchid`, `skunk`, `hamster`, `oak_tree`, `lizard`, `bridge`, `sunflower`, `pickup_truck`, `orange`, `man`, `mouse`, `cup`, `whale`, `seal`, `television`, `snake`, `crocodile`, `cockroach`, `bed`, `otter`, `caterpillar`, `woman`, `rocket`, `butterfly`, `bicycle`, `spider`, `motorcycle`, `lawn_mower`, `wolf`, `raccoon`, `can`, `cloud`, `clock`, `worm`, `tank`, `ray`, `house`, `girl`, `dinosaur`, `willow_tree`, `maple_tree`, `kangaroo`, `cattle`, `bee`, `chair`, `aquarium_fish`, `shark`, `baby`, `tractor`, `sweet_pepper`, `plain`, `lamp`, `boy`, `telephone`, `mushroom`, `couch`, `apple`, `wardrobe`, `train`, `pine_tree`, `pear`, `road`, `mountain`, `castle`, `bowl`, `lobster`, `elephant`, `beetle`, `possum`, `forest`, `flatfish`, `poppy`, `fox`, `streetcar`, `chimpanzee`, `bottle`, `rose`, `rabbit`, `table`, `camel`


## Predicted Entities

`angular_leaf_spot`, `bean_rust`, `healthy`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_beans_demo_v5_en_4.2.1_3.0_1665535239138.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_base_beans_demo_v5_en_4.2.1_3.0_1665535239138.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_base_beans_demo_v5', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_base_beans_demo_v5", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_base_beans_demo_v5|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.2 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification