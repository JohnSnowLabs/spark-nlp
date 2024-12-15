---
layout: docs
header: true
seotitle:
title: Image Classification
permalink: docs/en/tasks/image_classification
key: docs-tasks-image-classification
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

**Image classification** is the process of assigning a **label** or **category** to an image based on its visual content. This task is fundamental in the field of computer vision and has numerous applications, from facial recognition to product classification in e-commerce. Spark NLP provides tools that make it easier to integrate image classification into your data pipelines, allowing for scalable, efficient image processing within the same framework.

By using image classification models, we can analyze and classify images into predefined categories based on patterns and features in the image data. Some common use cases include:

- Classifying product images into categories like **clothing**, **electronics**, **furniture**, etc.
- Recognizing objects in images, such as identifying animals, vehicles, or various types of landscapes.
- Detecting facial expressions and other human features for tasks like emotion analysis or identity verification.

## Picking a Model

When selecting a model for image classification, it’s essential to consider several factors that ensure optimal performance for your specific use case. Start by evaluating the **type of images** you are working with, such as grayscale vs. colored, high-resolution vs. low-resolution, or simple vs. complex visual patterns. Determine whether your task requires **binary classification** (e.g., cat vs. dog) or **multiclass classification** (e.g., classifying various animal species), as the right model choice depends on the complexity of the task.

Next, assess the **computational power** available to you. Complex models such as CNNs (Convolutional Neural Networks) can be resource-intensive but deliver highly accurate results. Simpler models may be sufficient for less demanding tasks. Ensure the model's **performance metrics** (accuracy, precision, recall) align with your project goals, and consider the **interpretability** of the model—more advanced models may be less interpretable but offer greater accuracy.

Explore a wide variety of image classification models on the [Spark NLP Models](https://sparknlp.org/models), where you can find pre-trained models suited for different tasks and datasets.

#### Recommended Models for Specific Image Classification Tasks
- **Object Detection:** For detecting objects in images, models such as [`image_classifier_vit_base_patch16_224`](https://sparknlp.org/2022/08/10/image_classifier_vit_base_patch16_224_en_3_0.html){:target="_blank"} can be used to detect objects across multiple categories.
- **Facial Expression Recognition:** Models like [`image_classifier_swin_swin_large_patch4_window12_384`](https://sparknlp.org/2023/03/23/pipeline_image_classifier_swin_swin_large_patch4_window12_384_en.html){:target="_blank"} are great for tasks that involve recognizing facial emotions.
- **Scene Classification:** To classify scenes into categories like **urban**, **rural**, or **forest**, models like [`image_classifier_vit_base_patch16_224`](https://sparknlp.org/2022/08/10/image_classifier_vit_base_patch16_224_en_3_0.html){:target="_blank"} can be applied effectively.

By carefully considering your data, task requirements, and available resources, you can make an informed decision and leverage the best models for your image classification needs.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Load image data into a DataFrame, discarding any invalid images
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("src/test/resources/image/")

# Image Assembler: Prepares image data for processing
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

# ViTForImageClassification: Pretrained Vision Transformer model for image classification
imageClassifier = ViTForImageClassification \
    .pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("class")

# Create a pipeline with image assembler and classifier stages
pipeline = Pipeline().setStages([imageAssembler, imageClassifier])

# Fit the pipeline on the image DataFrame and transform the data
pipelineDF = pipeline.fit(imageDF).transform(imageDF)

# Select and display the image file name and the classification result
pipelineDF \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result") \
  .show(truncate=False)

+-----------------+----------------------------------------------------------+
|image_name       |result                                                    |
+-----------------+----------------------------------------------------------+
|palace.JPEG      |[palace]                                                  |
|egyptian_cat.jpeg|[Egyptian cat]                                            |
|hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
|hen.JPEG         |[hen]                                                     |
|ostrich.JPEG     |[ostrich, Struthio camelus]                               |
|junco.JPEG       |[junco, snowbird]                                         |
|bluetick.jpg     |[bluetick]                                                |
|chihuahua.jpg    |[Chihuahua]                                               |
|tractor.JPEG     |[tractor]                                                 |
|ox.JPEG          |[ox]                                                      |
+-----------------+----------------------------------------------------------+
```
```scala
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

// Load image data into a DataFrame, discarding invalid images
val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

// Image Assembler: Prepares image data for further processing
val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

// Pretrained ViT model for image classification
val imageClassifier = ViTForImageClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")

// Create a pipeline with the image assembler and classifier stages
val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

// Fit the pipeline on the image DataFrame and apply transformations
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

// Select and display the image name and the classification result
pipelineDF
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result")
  .show(truncate = false)

+-----------------+----------------------------------------------------------+
|image_name       |result                                                    |
+-----------------+----------------------------------------------------------+
|palace.JPEG      |[palace]                                                  |
|egyptian_cat.jpeg|[Egyptian cat]                                            |
|hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
|hen.JPEG         |[hen]                                                     |
|ostrich.JPEG     |[ostrich, Struthio camelus]                               |
|junco.JPEG       |[junco, snowbird]                                         |
|bluetick.jpg     |[bluetick]                                                |
|chihuahua.jpg    |[Chihuahua]                                               |
|tractor.JPEG     |[tractor]                                                 |
|ox.JPEG          |[ox]                                                      |
+-----------------+----------------------------------------------------------+
```

## Try Real-Time Demos!

If you want to explore real-time image classification outputs, visit our interactive demos:

- **[Swin For Image Classification](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-SwinForImageClassification){:target="_blank"}**
- **[VisionEncoderDecoder For Image Captioning](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-VisionEncoderDecoderForImageCaptioning){:target="_blank"}**
- **[Object Detection & Scene Classification](https://nlp.johnsnowlabs.com/detect_objects_scenes){:target="_blank"}**
- **[ConvNext For Image Classification](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-ConvNextForImageClassification){:target="_blank"}**

## Useful Resources

To dive deeper into image classification using Spark NLP, check out these useful resources:

**Notebooks**
- *[Image Classification Notebooks in SparkNLP](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/image){:target="_blank"}*
- *[ViT for Image Classification with Transformers](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/19.Image_Classification.ipynb){:target="_blank"}*
