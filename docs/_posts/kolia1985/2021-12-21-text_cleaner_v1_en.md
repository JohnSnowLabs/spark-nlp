---
layout: model
title: Text cleaner v1
author: John Snow Labs
name: text_cleaner_v1
date: 2021-12-21
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Visual NLP 3.0.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Model for cleaning image with text. It is based on text detection model with extra post-processing.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/text_cleaner_v1_en_3.0.0_2.4_1640088709401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
cleaner = ImageTextCleaner \
            .pretrained("text_cleaner_v1", "en", "clinical/ocr") \
            .setInputCol("image") \
            .setOutputCol("cleaned_image") \
            .setMedianBlur(0) \
            .setSizeThreshold(1) \
            .setTextThreshold(0.3) \
            .setPadding(2) \
            .setBinarize(False)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_cleaner_v1|
|Type:|ocr|
|Compatibility:|Visual NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|77.1 MB|