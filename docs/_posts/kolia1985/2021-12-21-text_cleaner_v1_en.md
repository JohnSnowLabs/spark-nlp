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
[Open in Colab]([Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/TrainingNotebooks/tutorials/Certification_Trainings/others/SparkOcrImageCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/text_cleaner_v1_en_3.0.0_2.4_1640088709401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use
<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *
    
    imagePath = "path to image"
    image_df = spark.read.format("binaryFile").load(imagePath)


## How to use
<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *
    
    imagePath = "path to image"
    image_df = spark.read.format("binaryFile").load(imagePath)

    binary_to_image = BinaryToImage() 
    
    text_detector = ImageTextDetectorV2 \
        .pretrained("image_text_detector_v2", "en", "clinical/ocr") \
        .setInputCol("image") \
        .setOutputCol("text_regions") \
        .setWithRefiner(True) \
        .setSizeThreshold(10) \
        .setScoreThreshold(0.2) \
        .setTextThreshold(0.2) \
        .setLinkThreshold(0.3) \
        .setWidth(500)
    
    ocr = ImageToTextV2.pretrained("ocr_base_handwritten", "en", "clinical/ocr") \
        .setInputCols(["image", "text_regions"]) \
        .setGroupImages(True) \
        .setOutputCol("text")
    
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

    result = pipeline.transform(image_df)
    print(f"Detected text:\n{results.select('text').collect()[0].text}")
```
```scala

```
</div>


## Example

### Input:
![Screenshot](../../_examples_ocr/image4.png)

### Output:
![Screenshot](../../_examples_ocr/image4_out.png)

```bash
Detected text:
 

 

 

Sample specifications written by
 , BLEND CASING RECASING

- OLD GOLD STRAIGHT Tobacco Blend

Control for Sample No. 5030

Cigarettes:

OLD GOLD STRAIGHT

 

John H. M. Bohlken

FINAL FLAVOR MENTHOL FLAVOR

Tars and Nicotine, Taste Panel, Burning Time, Gas Phase Analysis,
Benzo (A) Pyrene Analyses — T/C -CF~ O.C S51: Fee -

Written by -- John H. M. Bohlken
Original to -Mr. C. L. Tucker, dr.
Copies to ---Dr. A. W. Spears

C

~
```


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