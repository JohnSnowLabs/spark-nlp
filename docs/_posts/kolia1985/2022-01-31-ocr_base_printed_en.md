---
layout: model
title: Оcr base for printed text
author: John Snow Labs
name: ocr_base_printed
date: 2022-01-31
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Spark NLP 3.3.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ocr base model for recognise printed text based on TrOcr architecture.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_base_printed_en_3.3.3_2.4_1643640900966.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

ocr = ImageToTextv2().pretrained("ocr_base_printed", "en", "clinical/ocr")
ocr.setInputCols(["image"])
ocr.setOutputCol("text")

result = ocr.transform(image_text_lines_df).collect()
print(result[0].text)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ocr = ImageToTextv2().pretrained("ocr_base_printed", "en", "clinical/ocr")
ocr.setInputCols(["image"])
ocr.setOutputCol("text")

result = ocr.transform(image_text_lines_df).collect()
print(result[0].text)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ocr_base_printed|
|Type:|ocr|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|779.8 MB|