---
layout: model
title: English image_classifier_vit_CarViT ViTForImageClassification from abdusahmbzuai
author: John Snow Labs
name: image_classifier_vit_CarViT
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: ViTForImageClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_CarViT` is a English model originally trained by abdusahmbzuai.


## Predicted Entities

`Toyota`, `Audi`, `Dodge`, `Aston Martin`, `Chevrolet`, `Mitsubishi`, `Kia`, `Honda`, `Chrysler`, `Lexus`, `Land Rover`, `Rolls-Royce`, `Porsche`, `FIAT`, `Cadillac`, `Jaguar`, `smart`, `Tesla`, `Maserati`, `Buick`, `GMC`, `Genesis`, `McLaren`, `Bentley`, `BMW`, `Lincoln`, `Subaru`, `Volvo`, `Lamborghini`, `Nissan`, `Alfa Romeo`, `Jeep`, `INFINITI`, `Mazda`, `Hyundai`, `Volkswagen`, `Ram`, `Ferrari`, `Acura`, `Mercedes-Benz`, `MINI`, `Ford`



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_CarViT_en_4.1.0_3.0_1660165745338.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_vit_CarViT_en_4.1.0_3.0_1660165745338.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_CarViT", "en")\
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

val imageAssembler = new ImageAssembler()
.setInputCol("image")
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification
.pretrained("image_classifier_vit_CarViT", "en")
.setInputCols("image_assembler")
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```


{:.nlu-block}
```python
import nlu
import requests
response = requests.get('https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/docs/assets/images/hen.JPEG')
with open('hen.JPEG', 'wb') as f:
    f.write(response.content)
nlu.load("en.classify_image.CarViT").predict("hen.JPEG")
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_CarViT|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|322.0 MB|
