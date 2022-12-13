---
layout: model
title: English image_classifier_vit_shirt_identifier ViTForImageClassification from b25mayank3
author: John Snow Labs
name: image_classifier_vit_shirt_identifier
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_shirt_identifier` is a English model originally trained by b25mayank3.


## Predicted Entities

`Big Check shirt`, `Formal Shirt`, `casual shirt`, `denim shirt`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_shirt_identifier_en_4.1.0_3.0_1660171643901.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_vit_shirt_identifier_en_4.1.0_3.0_1660171643901.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_shirt_identifier", "en")\
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
.pretrained("image_classifier_vit_shirt_identifier", "en")\
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
|Model Name:|image_classifier_vit_shirt_identifier|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|