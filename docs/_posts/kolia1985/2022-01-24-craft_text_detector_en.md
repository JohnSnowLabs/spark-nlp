---
layout: model
title: Craft text detection with refiner
author: John Snow Labs
name: craft_text_detector
date: 2022-01-24
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Spark NLP 3.4.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Craft text detection with refiner

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/craft_text_detector_en_3.4.0_2.4_1643026460495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

text_detector = Craft().pretrained("сraft_text_detector", "en", "clinical/ocr") \
     .setInputCol("image") \
     .setOutputCol("text_regions")

 draw_regions = ImageDrawRegions()
 draw_regions.setInputCol("image")
 draw_regions.setInputRegionsCol("text_regions")
 draw_regions.setOutputCol("image_with_regions")
 draw_regions.setFilledRect(False)
 draw_regions.setRectColor(Color.black)
 draw_regions.setRotated(True)

 result = draw_regions.transform(text_detector.transform(image_df)).cache()
 display_images_horizontal(result, "image_with_regions", width=300)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
text_detector = Craft().pretrained("сraft_text_detector", "en", "clinical/ocr") \
     .setInputCol("image") \
     .setOutputCol("text_regions")

 draw_regions = ImageDrawRegions()
 draw_regions.setInputCol("image")
 draw_regions.setInputRegionsCol("text_regions")
 draw_regions.setOutputCol("image_with_regions")
 draw_regions.setFilledRect(False)
 draw_regions.setRectColor(Color.black)
 draw_regions.setRotated(True)

 result = draw_regions.transform(text_detector.transform(image_df)).cache()
 display_images_horizontal(result, "image_with_regions", width=300)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|craft_text_detector|
|Type:|ocr|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|79.0 MB|