---
layout: model
title: General model for table detection
author: John Snow Labs
name: ocr_table_detection_general_model
date: 2021-09-04
tags: [en, licensed]
task: OCR Table Detection & Recognition
language: en
edition: Visual NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

General model for table detection inspired by https://arxiv.org/abs/2004.12629

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_table_detection_general_model_en_3.0.0_3.0_1630757579641.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

This modes is used by ImageTableDetector

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
binary_to_image = BinaryToImage()
binary_to_image.setImageType(ImageType.TYPE_3BYTE_BGR)

table_detector = ImageTableDetector
.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
.setInputCol("image")
.setOutputCol("table_regions")

pipeline = PipelineModel(stages=[
    binary_to_image,
    table_detector
])
```
```scala
var imgDf = spark.read.format("binaryFile").load(imagePath)
var bin2imTransformer = new BinaryToImage()
bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

val dataFrame = bin2imTransformer.transform(imgDf)
val tableDetector = ImageTableDetector
.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
.setInputCol("image")
.setOutputCol("table regions")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ocr_table_detection_general_model|
|Type:|ocr|
|Compatibility:|Visual NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[table regions]|
|Language:|en|
