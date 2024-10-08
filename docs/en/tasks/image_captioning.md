---
layout: docs
header: true
seotitle:
title: Image Captioning
permalink: docs/en/tasks/image_captioning
key: docs-tasks-image-captioning
modify_date: "2024-10-05"
show_nav: true
sidebar:
  nav: sparknlp
---

**Image Captioning** is the process of generating descriptive text for an image based on its visual content. This task is crucial in computer vision and has a variety of applications, such as enhancing accessibility for visually impaired individuals, improving image search, and enriching multimedia content. Spark NLP integrates image captioning with other NLP and vision-based tasks, enabling efficient and scalable caption generation within the same framework.

By utilizing image captioning models, we can produce natural language descriptions that capture the key elements and context of images. Common use cases include:

- **Social Media**: Automatically generating captions for user-uploaded images.
- **E-Commerce**: Generating product descriptions based on visual attributes.
- **Accessibility**: Describing visual content for the visually impaired.
- **Search Engines**: Improving search results by associating images with relevant text.

## Picking a Model

When selecting a model for image captioning, it’s important to consider the **image complexity** and the **quality of captions** required. For example, some tasks may need simple, high-level descriptions (e.g., "a person riding a bike"), while others might require more detailed, context-rich captions (e.g., "a young man riding a mountain bike on a sunny day").

Additionally, assess the **performance metrics** such as **BLEU score** or **ROUGE score** for evaluating the quality of generated captions. Ensure that the model is well-suited to your specific dataset, whether it consists of simple images like products or more complex images like natural scenes.

Explore pre-trained image captioning models in the [Spark NLP Models Hub](https://sparknlp.org/models) for a variety of datasets and tasks.

#### Recommended Models for Image Captioning
- **VisionEncoderDecoder For Image Captioning:** This model can be used for generating descriptive captions based on images. It utilizes a transformer-based architecture, providing high-quality captions for various types of images. Check out the pre-trained model [`image-captioning-vit-gpt2`](https://sparknlp.org/2023/09/20/image_captioning_vit_gpt2_en.html){:target="_blank"}.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
# Import necessary libraries from Spark NLP
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Load image data into a DataFrame, discarding any invalid images
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("src/test/resources/image/")

# Create an ImageAssembler to prepare image data for processing
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

# Initialize the VisionEncoderDecoder model for image captioning
imageCaptioning = VisionEncoderDecoderForImageCaptioning \
    .pretrained() \  # Load a pre-trained model for image captioning
    .setBeamSize(2)
    .setDoSample(False)
    .setInputCols(["image_assembler"]) \
    .setOutputCol("caption")

# Create a pipeline that includes the image assembler and image captioning stages
pipeline = Pipeline().setStages([imageAssembler, imageCaptioning])

# Fit the pipeline on the image DataFrame and transform the data
pipelineDF = pipeline.fit(imageDF).transform(imageDF)

# Select and display the image file name and the generated captions
pipelineDF \
    .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result") \
    .show(truncate=False)

+-----------------+---------------------------------------------------------+
|image_name       |result                                                   |
+-----------------+---------------------------------------------------------+
|palace.JPEG      |[a large room filled with furniture and a large window]  |
|egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
|hippopotamus.JPEG|[a brown bear in a body of water]                        |
|hen.JPEG         |[a flock of chickens standing next to each other]        |
|ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
|junco.JPEG       |[a small bird standing on a wet ground]                  |
|bluetick.jpg     |[a small dog standing on a wooden floor]                 |
|chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
|tractor.JPEG     |[a man is standing in a field with a tractor]            |
|ox.JPEG          |[a large brown cow standing on top of a lush green field]|
+-----------------+---------------------------------------------------------+
```
```scala
// Import necessary libraries from Spark NLP
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

// Load image data into a DataFrame, discarding invalid images
val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

// Image Assembler: Prepares image data for processing
val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

// Initialize image captioning model
val imageCaptioning = VisionEncoderDecoderForImageCaptioning
  .pretrained()
  .setBeamSize(2)
  .setDoSample(false)
  .setInputCols("image_assembler")
  .setOutputCol("caption")

// Create and fit the pipeline
val pipeline = new Pipeline().setStages(Array(imageAssembler, imageCaptioning))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

// Display image names and generated captions
pipelineDF
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")
  .show(truncate = false)

+-----------------+---------------------------------------------------------+
|image_name       |result                                                   |
+-----------------+---------------------------------------------------------+
|palace.JPEG      |[a large room filled with furniture and a large window]  |
|egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
|hippopotamus.JPEG|[a brown bear in a body of water]                        |
|hen.JPEG         |[a flock of chickens standing next to each other]        |
|ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
|junco.JPEG       |[a small bird standing on a wet ground]                  |
|bluetick.jpg     |[a small dog standing on a wooden floor]                 |
|chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
|tractor.JPEG     |[a man is standing in a field with a tractor]            |
|ox.JPEG          |[a large brown cow standing on top of a lush green field]|
+-----------------+---------------------------------------------------------+
```

## Try Real-Time Demos!

Explore real-time image captioning outputs with our interactive demos:

- **[VisionEncoderDecoder For Image Captioning](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-VisionEncoderDecoderForImageCaptioning){:target="_blank"}**

## Useful Resources

To dive deeper into image captioning using Spark NLP, check out these useful resources:

**Notebooks**
- *[Vision Encoder-Decoder for Image Captioning](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/VisionEncoderDecoderForImageCaptioning.ipynb){:target="_blank"}*
