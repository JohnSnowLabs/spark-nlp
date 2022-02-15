---
layout: model
title: Image Text Detector V2
author: John Snow Labs
name: image_text_detector_v2
date: 2022-01-31
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Image Text Detector based on the CRAFT architecture with refiner net.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/image_text_detector_v2_en_3.3.0_2.4_1643618928538.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

text_detector = ImageTextDetectorV2().pretrained("image_text_detector_v2", "en", "clinical/ocr") \
     .setInputCol("image") \
     .setOutputCol("text_regions")

 draw_regions = ImageDrawRegions()
 draw_regions.setInputCol("image")
 draw_regions.setInputRegionsCol("text_regions")
 draw_regions.setOutputCol("image_with_regions")
 draw_regions.setFilledRect(False)
 draw_regions.setRectColor(Color.black)
 draw_regions.setRotated(True)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
text_detector = ImageTextDetectorV2().pretrained("image_text_detector_v2", "en", "clinical/ocr") \
     .setInputCol("image") \
     .setOutputCol("text_regions")

 draw_regions = ImageDrawRegions()
 draw_regions.setInputCol("image")
 draw_regions.setInputRegionsCol("text_regions")
 draw_regions.setOutputCol("image_with_regions")
 draw_regions.setFilledRect(False)
 draw_regions.setRectColor(Color.black)
 draw_regions.setRotated(True)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_text_detector_v2|
|Type:|ocr|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|79.0 MB|