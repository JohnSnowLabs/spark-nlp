---
layout: model
title: Image Processing algorithms to improve Document Quality
author: John Snow Labs
name: image_processing
date: 2023-01-03
tags: [en, licensed, ocr, image_processing]
task: Document Image Processing
language: en
nav_key: models
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
binary_to_image = BinaryToImage() \
    .setInputCol("content")  \
    .setOutputCol("image") 

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

pipeline_scaled = PipelineModel(stages=[
    binary_to_image,
    scaled_image_df
])

pipeline_thresholded = PipelineModel(stages=[
    binary_to_image,
    thresholded_image
])

pipeline_eroded = PipelineModel(stages=[
    binary_to_image,
    eroded_image
])

pipeline_dilated = PipelineModel(stages=[
    binary_to_image,
    dilated_image
])

pipeline_removebg = PipelineModel(stages=[
    binary_to_image,
    removebg_image
])

pipeline_deblured = PipelineModel(stages=[
    binary_to_image,
    deblured_image
])

pipeline_multiple = PipelineModel(stages=[
    binary_to_image,
    multiple_image
])

image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/images/check.jpg")
image_example_df = spark.read.format("binaryFile").load(image_path)

result_scaled = pipeline_scaled.transform(image_example_df).cache()
result_thresholded = pipeline_thresholded.transform(image_example_df).cache()
result_eroded = pipeline_eroded.transform(image_example_df).cache()
result_dilated = pipeline_dilated.transform(image_example_df).cache()
result_removebg = pipeline_removebg.transform(image_example_df).cache()
result_deblured = pipeline_deblured.transform(image_example_df).cache()
result_multiple = pipeline_multiple.transform(image_example_df).cache()
```
```scala
val binary_to_image = new BinaryToImage() 
    .setInputCol("content")  
    .setOutputCol("image") 

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

val pipeline_scaled = new PipelineModel().setStages(Array(
    binary_to_image, 
    scaled_image_df))
    
val pipeline_thresholded = new PipelineModel().setStages(Array(
    binary_to_image, 
    thresholded_image))
    
val pipeline_eroded = new PipelineModel().setStages(Array(
    binary_to_image, 
    eroded_image))
    
val pipeline_dilated = new PipelineModel().setStages(Array(
    binary_to_image, 
    dilated_image))
    
val pipeline_removebg = new PipelineModel().setStages(Array(
    binary_to_image, 
    removebg_image))
    
val pipeline_deblured = new PipelineModel().setStages(Array(
    binary_to_image, 
    deblured_image))
    
val pipeline_multiple = new PipelineModel().setStages(Array(
    binary_to_image, 
    multiple_image))

val image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/images/check.jpg")
val image_example_df = spark.read.format("binaryFile").load(image_path)

val result_scaled = pipeline_scaled.transform(image_example_df).cache()
val result_thresholded = pipeline_thresholded.transform(image_example_df).cache()
val result_eroded = pipeline_eroded.transform(image_example_df).cache()
val result_dilated = pipeline_dilated.transform(image_example_df).cache()
val result_removebg = pipeline_removebg.transform(image_example_df).cache()
val result_deblured = pipeline_deblured.transform(image_example_df).cache()
val result_multiple = pipeline_multiple.transform(image_example_df).cache()
```
</div>

## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image2.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image2_out2.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}