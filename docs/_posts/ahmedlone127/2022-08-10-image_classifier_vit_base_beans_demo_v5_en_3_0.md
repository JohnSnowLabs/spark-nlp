---
layout: model
title: English image_classifier_vit_base_beans_demo_v5 ViTForImageClassification from Miss
author: John Snow Labs
name: image_classifier_vit_base_beans_demo_v5
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_base_beans_demo_v5` is a English model originally trained by Miss.


## Predicted Entities

`lion`, `tulip`, `keyboard`, `cra`, `bus`, `dolphin`, `plate`, `beaver`, `skyscraper`, `tiger`, `bear`, `trout`, `porcupine`, `sea`, `shrew`, `squirrel`, `snail`, `leopard`, `palm_tree`, `turtle`, `orchid`, `skunk`, `hamster`, `oak_tree`, `lizard`, `bridge`, `sunflower`, `pickup_truck`, `orange`, `man`, `mouse`, `cup`, `whale`, `seal`, `television`, `snake`, `crocodile`, `cockroach`, `bed`, `otter`, `caterpillar`, `woman`, `rocket`, `butterfly`, `bicycle`, `spider`, `motorcycle`, `lawn_mower`, `wolf`, `raccoon`, `can`, `cloud`, `clock`, `worm`, `tank`, `ray`, `house`, `girl`, `dinosaur`, `willow_tree`, `maple_tree`, `kangaroo`, `cattle`, `bee`, `chair`, `aquarium_fish`, `shark`, `baby`, `tractor`, `sweet_pepper`, `plain`, `lamp`, `boy`, `telephone`, `mushroom`, `couch`, `apple`, `wardrobe`, `train`, `pine_tree`, `pear`, `road`, `mountain`, `castle`, `bowl`, `lobster`, `elephant`, `beetle`, `possum`, `forest`, `flatfish`, `poppy`, `fox`, `streetcar`, `chimpanzee`, `bottle`, `rose`, `rabbit`, `table`, `camel`


## Predicted Entities

`angular_leaf_spot`, `bean_rust`, `healthy`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_base_beans_demo_v5_en_4.1.0_3.0_1660170085775.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_base_beans_demo_v5", "en")\
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
.pretrained("image_classifier_vit_base_beans_demo_v5", "en")\
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
|Model Name:|image_classifier_vit_base_beans_demo_v5|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.2 MB|