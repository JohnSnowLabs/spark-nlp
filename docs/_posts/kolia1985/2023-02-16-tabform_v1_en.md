---
layout: model
title: TabForm v1
author: John Snow Labs
name: tabform_v1
date: 2023-02-16
tags: [en, licensed]
task: OCR Table Detection & Recognition
language: en
edition: Visual NLP 4.2.5
spark_version: 3.2
supported: true
annotator: ImageTableDetector
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Tabform is an optimized model that utilizes YOLO to accurately separate tables and forms. The model was specifically trained on a large dataset containing tables and forms, including Internal Revenue Service (IRS) tax forms, which are used to report financial information to the government. IRS is a large dataset containing various forms and tables used for tax reporting in the United States. It is likely that the dataset includes both structured and unstructured data, such as text, numerical data, and images of the forms. YOLO is a compound-scaled object detection model trained on the COCO dataset. It was introduced as the first object detection model to combine bounding box prediction and object classification into a single end-to-end differentiable network. While closely related to image classification, object detection performs image classification on a more precise scale by locating and categorizing features in images. Overall, Tabform model’s accuracy for the test set is 98.5%, making it a highly effective tool for separating tables and forms.

## Predicted Entities

`table`, `form`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/tabform_v1_en_4.2.5_3.2_1676540271876.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/tabform_v1_en_4.2.5_3.2_1676540271876.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

table_detector = ImageTableDetector \
            .pretrained("tabform_v1", "en", "public/ocr/models") \
            .setInputCol("image") \
            .setOutputCol("table_regions") \
            .setScoreThreshold(0.9) \
            .setApplyCorrection(True)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
table_detector = ImageTableDetector \
            .pretrained("tabform_v1", "en", "public/ocr/models") \
            .setInputCol("image") \
            .setOutputCol("table_regions") \
            .setScoreThreshold(0.9) \
            .setApplyCorrection(True)
```
```scala
val tableDetector = ImageTableDetector
    .pretrained("tabform_v1", "en", "public/ocr/models")
    .setInputCol("image")
    .setOutputCol("table regions")
    .setApplyCorrection(true)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tabform_v1|
|Type:|ocr|
|Compatibility:|Visual NLP 4.2.5+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[table_regions]|
|Language:|en|
|Size:|24.2 MB|