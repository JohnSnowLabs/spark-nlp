---
layout: model
title: Оcr base v2 for printed text
author: John Snow Labs
name: ocr_base_printed_v2
date: 2023-01-17
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Visual NLP 4.2.4
spark_version: 3.2.1
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ocr base printed model v2 for recognise printed text based on Tesseract architecture and printed datasets.
Tesseract is an optical character recognition engine with open-source code, this is the most popular and qualitative OCR-library. OCR uses artificial intelligence for text search and its recognition on images. Tesseract is finding templates in pixels, letters, words and sentences. It uses two-step approach that calls adaptive recognition. It requires one data stage for character recognition, then the second stage to fulfil any letters, it wasn’t insured in, by letters that can match the word or sentence context.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Cards/SparkOcrImageToTextPrinted_V2.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_base_printed_v2_en_4.2.2_3.0_1670623909000.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *

    imagePath = "path to image"
    image_df = spark.read.format("binaryFile").load(imagePath)

    binary_to_image = BinaryToImage() 
    binary_to_image.setImageType(ImageType.TYPE_3BYTE_BGR)
    
    text_detector = ImageTextDetectorV2 \
        .pretrained("image_text_detector_v2", "en", "clinical/ocr") \
        .setInputCol("image") \
        .setOutputCol("text_regions") \
        .setWithRefiner(True) \
        .setSizeThreshold(-1) \
        .setLinkThreshold(0.3) \
        .setWidth(500)
    
    ocr = ImageToTextV2Opt.pretrained("ocr_base_printed_v2", "en", "clinical/ocr") \
        .setInputCols(["image", "text_regions"]) \
        .setGroupImages(True) \
        .setOutputCol("text") \
        .setRegionsColumn("text_regions")
    
    draw_regions = ImageDrawRegions() \
        .setInputCol("image") \
        .setInputRegionsCol("text_regions") \
        .setOutputCol("image_with_regions") \
        .setRectColor(Color.green) \
        .setRotated(True)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        text_detector,
        ocr,
        draw_regions
    ])

    result = pipeline.transform(image_df).cache()
    display_images(result, "image_with_regions")
    print(("").join([x.text for x in result.select("text").collect()]))
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"
val imageDf = spark.read.format("binaryFile").load(imagePath)

val regionsPath = "./python/sparkocr/resources/ocr/text_detection/regions.parquet"
val regionsDf = spark.read.parquet(regionsPath)

val imageWithRegions = regionsDf.join(imageDf)

val binaryToImage = new BinaryToImage().
  setOutputCol("image")

val ocr = ImageToTextOnnx.
  pretrained("ocr_base_printed_v2")
   .setInputCols(Array("image"))
   .setRegionsColumn("text_regions")
   .setOutputFormat("text")

val pipeline = new Pipeline()
   .setStages(Array(binaryToImage, ocr))
  .fit(imageWithRegions)

val r = Benchmark.time("Using ocr_base_printed_v2",true) {
  pipeline.transform(imageWithRegions).select("text").collect()
}
assert(r.head.getString(0).contains("performance"))
assert(r.head.getString(0).contains("printed"))
assert(r.head.getString(0).contains("hope"))
assert(r.head.getString(0).length > 90)
```
</div>

## Example

### Input:
![Screenshot](../../_examples_ocr/image2.png)

### Output:
![Screenshot](../../_examples_ocr/image2_out3.png)
```bash
STARBUCKS STORE #10208
11302 EUCLID AVENUE
CLEVELAND, OH (216) 229-0749
CHK 664290
12/07/2014 06:43 PM
1912003- DRAWER: 2. REG: 2
VT PEP MOCHA
SBUX CARD : 4.95
XXXXXXXXXXXX3228
SUBTOTAL: $4.95
TOTAL @ 6.95
CHANGE DUE $0.00
................ CCHECK CLOSED
12/07/2014 06:43 PM
SBUX CARD X3228 NEW BALANCE: 37.45
CARD IS REGISTERED
```



