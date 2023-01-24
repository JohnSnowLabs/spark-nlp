---
layout: model
title: Image Processing algorithms to improve Document Quality
author: John Snow Labs
name: image_processing
date: 2023-01-03
tags: [en, licensed, ocr, image_processing]
task: Document Image Processing
language: en
edition: Visual NLP 4.0.0
spark_version: 3.2.1
supported: true
annotator: ImageProcessing
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The processing of documents for the purpose of discovering  knowledge from them in an automated fashion is a challenging task and hence an open issue for the research community. Sometimes, the quality of the input images makes it much more difficult to perform these procedures correctly.

To avoid this, it is proposed the use of different image processing algorithms on documents images to improve its quality and the performance of the next step of computer vision algorithms as text detection, text recognition, ocr, table detection... Some of these image processing algorithms included in this project are: scale image, adaptive thresholding, erosion, dilation, remove objects, median blur and gpu image transformation.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/1.2.Image_processing.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/images/check.jpg")
image_example_df = spark.read.format("binaryFile").load(image_path)
image_df = BinaryToImage().transform(image_example_df).cache()

scaled_image_df = ImageTransformer() \
    .addScalingTransform(2) \
    .setInputCol("image") \
    .setOutputCol("scaled_image") \
    .setImageType(ImageType.TYPE_BYTE_GRAY) \
    .transform(image_df)

thresholded_image = ImageTransformer() \
    .addAdaptiveThreshold(21, 20)\
    .setInputCol("image") \
    .setOutputCol("thresholded_image") \
    .transform(image_df)

eroded_image = ImageTransformer() \
    .addErodeTransform(2,2)\
    .setInputCol("image") \
    .setOutputCol("eroded_image") \
    .transform(image_df)

dilated_image = ImageTransformer() \
    .addDilateTransform(1, 2)\
    .setInputCol("image") \
    .setOutputCol("dilated_image") \
    .transform(image_df)

removebg_image = ImageTransformer() \
    .addScalingTransform(2) \
    .addAdaptiveThreshold(31, 2)\
    .addRemoveObjects(10, 500) \
    .setInputCol("image") \
    .setOutputCol("corrected_image") \
    .transform(image_df)

deblured_image = ImageTransformer() \
    .addScalingTransform(2) \
    .addMedianBlur(3) \
    .setInputCol("image") \
    .setOutputCol("corrected_image") \
    .transform(image_df)

multiple_image = GPUImageTransformer() \
    .addScalingTransform(8) \
    .addOtsuTransform() \
    .addErodeTransform(3, 3) \
    .setInputCol("image") \
    .setOutputCol("multiple_image") \
    .transform(image_df)
```
```scala
val image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/images/check.jpg")
val image_example_df = spark.read.format("binaryFile").load(image_path)
val image_df = new BinaryToImage().transform(image_example_df).cache()

val scaled_image_df = new ImageTransformer()
    .addScalingTransform(2) 
    .setInputCol("image") 
    .setOutputCol("scaled_image") 
    .setImageType(ImageType.TYPE_BYTE_GRAY) 
    .transform(image_df)

val thresholded_image = new ImageTransformer() 
    .addAdaptiveThreshold(21, 20)
    .setInputCol("image") 
    .setOutputCol("thresholded_image") 
    .transform(image_df)

val eroded_image = new ImageTransformer() 
    .addErodeTransform(2,2)
    .setInputCol("image") 
    .setOutputCol("eroded_image") 
    .transform(image_df)

val dilated_image = new ImageTransformer() 
    .addDilateTransform(1, 2)
    .setInputCol("image") 
    .setOutputCol("dilated_image") 
    .transform(image_df)

val removebg_image = new ImageTransformer() 
    .addScalingTransform(2) 
    .addAdaptiveThreshold(31, 2)
    .addRemoveObjects(10, 500) 
    .setInputCol("image") 
    .setOutputCol("corrected_image") 
    .transform(image_df)

val deblured_image = new ImageTransformer() 
    .addScalingTransform(2) 
    .addMedianBlur(3) 
    .setInputCol("image") 
    .setOutputCol("corrected_image") 
    .transform(image_df)

val multiple_image = new GPUImageTransformer() 
    .addScalingTransform(8) 
    .addOtsuTransform() 
    .addErodeTransform(3, 3) 
    .setInputCol("image") 
    .setOutputCol("multiple_image") 
    .transform(image_df)
```
</div>

## Example

### Input:
![Screenshot](../../_examples_ocr/image2.png)

### Output:
![Screenshot](../../_examples_ocr/image2_out2.png)

