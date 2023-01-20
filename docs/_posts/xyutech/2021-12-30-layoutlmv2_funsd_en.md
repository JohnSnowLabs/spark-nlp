---
layout: model
title: LayoutLMv2 model for document understanding
author: John Snow Labs
name: layoutlmv2_funsd
date: 2021-12-30
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Healthcare NLP 3.3.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained model for text and understanding. This model was trained on FUNSD dataset.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/layoutlmv2_funsd_en_3.3.0_3.0_1640854497540.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/layoutlmv2_funsd_en_3.3.0_3.0_1640854497540.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
bin_to_image = BinaryToImage()\
    .setOutputCol("image")

ocr = ImageToHocr()\
    .setInputCol("image")\
    .setOutputCol("hocr")\
    .setIgnoreResolution(False)\
    .setOcrParams(["preserve_interword_spaces=0"])

tokenizer = HocrTokenizer()\
    .setInputCol("hocr")\
    .setOutputCol("token")

doc_ner = VisualDocumentNerV2()\
    .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")\
    .setInputCols(["token", "image"])\
    .setOutputCol("entities")

pipeline = PipelineModel(stages=[
    bin_to_image,
    ocr,
    tokenizer,
    doc_ner
    ])
```
```scala
var bin2imTransformer = new BinaryToImage()
bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

val ocr = new ImageToHocr()
    .setInputCol("image")
    .setOutputCol("hocr")
    .setIgnoreResolution(false)
    .setOcrParams(Array("preserve_interword_spaces=0"))

val tokenizer = new HocrTokenizer()
    .setInputCol("hocr")
    .setOutputCol("token")

val visualDocumentNER = VisualDocumentNERv2
    .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")
    .setInputCols(Array("token", "image"))

val pipeline = new Pipeline()
    .setStages(Array(
        bin2imTransformer,
        ocr,
        tokenizer,
        visualDocumentNER
    )
)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|layoutlmv2_funsd|
|Type:|ocr|
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|735.8 MB|

## Data Source

https://guillaumejaume.github.io/FUNSD/