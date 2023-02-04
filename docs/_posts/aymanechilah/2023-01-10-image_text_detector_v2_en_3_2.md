---
layout: model
title: Text Detection
author: John Snow Labs
name: image_text_detector_v2
date: 2023-01-10
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Visual NLP 4.1.0
spark_version: 3.2.1
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/Cards/SparkOcrImageTextDetection.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/image_text_detector_v2_en_3.3.0_2.4_1643618928538.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

text_detector = ImageTextDetectorV2 \
    .pretrained("image_text_detector_v2", "en", "clinical/ocr") \
    .setInputCol("image") \
    .setOutputCol("text_regions") \
    .setScoreThreshold(0.5) \
    .setTextThreshold(0.2) \
    .setSizeThreshold(10) \
    .setWithRefiner(True)

draw_regions = ImageDrawRegions() \
    .setInputCol("image") \
    .setInputRegionsCol("text_regions") \
    .setOutputCol("image_with_regions") \
    .setRectColor(Color.green) \
    .setRotated(True)

pipeline = PipelineModel(stages=[
    binary_to_image,
    text_detector,
    draw_regions
])

imagePath = pkg_resources.resource_filename('sparkocr', 'resources/ocr/text_detection/020_Yas_patella1.jpg')
image_df = spark.read.format("binaryFile").load(imagePath).sort("path")

result = pipeline.transform(image_df)
```
```scala
val binary_to_image = new BinaryToImage() 
    .setInputCol("content") 
    .setOutputCol("image") 
    .setImageType(ImageType.TYPE_3BYTE_BGR)

val text_detector = ImageTextDetectorV2 
    .pretrained("image_text_detector_v2", "en", "clinical/ocr") 
    .setInputCol("image") 
    .setOutputCol("text_regions") 
    .setScoreThreshold(0.5) 
    .setTextThreshold(0.2) 
    .setSizeThreshold(10) 
    .setWithRefiner(True)

val draw_regions = new ImageDrawRegions() 
    .setInputCol("image") 
    .setInputRegionsCol("text_regions") 
    .setOutputCol("image_with_regions") 
    .setRectColor(Color.green) 
    .setRotated(True)

val pipeline = new PipelineModel().setStages(Array(
    binary_to_image, 
    text_detector, 
    draw_regions))

val imagePath = pkg_resources.resource_filename("sparkocr", "resources/ocr/text_detection/020_Yas_patella1.jpg")
val image_df = spark.read.format("binaryFile").load(imagePath).sort("path")

val result = pipeline.transform(image_df)
```

</div>


## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image6.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image6_out.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_detection_v2|
|Type:|ocr|
|Compatibility:|Visual NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[text_regions]|
|Language:|en|

