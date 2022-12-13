---
layout: model
title: English image_classifier_vit_modeversion1_m6_e4n ViTForImageClassification from sudo-s
author: John Snow Labs
name: image_classifier_vit_modeversion1_m6_e4n
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_modeversion1_m6_e4n` is a English model originally trained by sudo-s.


## Predicted Entities

`45`, `98`, `113`, `34`, `67`, `120`, `93`, `142`, `147`, `12`, `66`, `89`, `51`, `124`, `84`, `8`, `73`, `78`, `19`, `100`, `23`, `62`, `135`, `128`, `4`, `121`, `88`, `77`, `40`, `110`, `15`, `11`, `104`, `90`, `9`, `141`, `139`, `132`, `44`, `33`, `117`, `22`, `56`, `55`, `26`, `134`, `50`, `123`, `37`, `68`, `61`, `107`, `13`, `46`, `99`, `24`, `94`, `83`, `35`, `16`, `79`, `5`, `103`, `112`, `72`, `10`, `59`, `144`, `87`, `48`, `21`, `116`, `76`, `138`, `54`, `43`, `148`, `127`, `65`, `71`, `57`, `108`, `32`, `80`, `106`, `137`, `82`, `49`, `6`, `126`, `36`, `1`, `39`, `140`, `17`, `25`, `60`, `14`, `133`, `47`, `122`, `111`, `102`, `31`, `96`, `69`, `95`, `58`, `145`, `64`, `53`, `42`, `75`, `115`, `109`, `0`, `20`, `27`, `70`, `2`, `86`, `38`, `81`, `118`, `92`, `125`, `18`, `101`, `30`, `7`, `143`, `97`, `130`, `114`, `129`, `29`, `41`, `105`, `63`, `3`, `74`, `91`, `52`, `85`, `131`, `28`, `119`, `136`, `146`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_modeversion1_m6_e4n_en_4.1.0_3.0_1660168388194.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_vit_modeversion1_m6_e4n_en_4.1.0_3.0_1660168388194.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_modeversion1_m6_e4n", "en")\
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
.pretrained("image_classifier_vit_modeversion1_m6_e4n", "en")\
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
|Model Name:|image_classifier_vit_modeversion1_m6_e4n|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.3 MB|