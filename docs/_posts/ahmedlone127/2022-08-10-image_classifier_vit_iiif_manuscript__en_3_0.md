---
layout: model
title: English image_classifier_vit_iiif_manuscript_ ViTForImageClassification from davanstrien
author: John Snow Labs
name: image_classifier_vit_iiif_manuscript_
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_iiif_manuscript_` is a English model originally trained by davanstrien.


## Predicted Entities

`3rd upper flyleaf verso`, `Blank leaf recto`, `3rd lower flyleaf verso`, `2nd lower flyleaf verso`, `2nd upper flyleaf verso`, `flyleaf`, `1st upper flyleaf verso`, `1st lower flyleaf verso`, `fol`, `cover`, `Lower flyleaf verso`, `Blank leaf verso`, `Upper flyleaf verso`



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_iiif_manuscript__en_4.1.0_3.0_1660170321999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_vit_iiif_manuscript__en_4.1.0_3.0_1660170321999.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_iiif_manuscript_", "en")\
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
.pretrained("image_classifier_vit_iiif_manuscript_", "en")
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
nlu.load("en.classify_image.iiif_manuscript_").predict("hen.JPEG")
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_iiif_manuscript_|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|
