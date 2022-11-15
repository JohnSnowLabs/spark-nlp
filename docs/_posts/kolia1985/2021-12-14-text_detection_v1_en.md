---
layout: model
title: Text Detection
author: John Snow Labs
name: text_detection_v1
date: 2021-12-14
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

CRAFT: Character-Region Awareness For Text detection, is designed with a convolutional neural network producing the character region score and affinity score. The region score is used to localize individual characters in the image, and the affinity score is used to group each character into a single instance. To compensate for the lack of character-level annotations, we propose a weaklysupervised learning framework that estimates characterlevel ground truths in existing real word-level datasets.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/text_detection_v1_en_3.0.0_2.4_1639490832988.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

text_detector = ImageTextDetector.pretrained("text_detection_v1", "en", "clinical/ocr")
text_detector.setInputCol("image")
text_detector.setOutputCol("text_regions")

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
text_detector = ImageTextDetector.pretrained("text_detection_v1", "en", "clinical/ocr")
text_detector.setInputCol("image")
text_detector.setOutputCol("text_regions")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_detection_v1|
|Type:|ocr|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[text_regions]|
|Language:|en|
|Size:|77.1 MB|