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

Image captioning is the task of automatically generating a short, meaningful description for an image. It combines **computer vision**, which helps the model understand what is in the image, with **natural language processing**, which enables it to express that understanding in human language. For example, given an image of a dog running on the beach, the model might generate a caption like *"A dog running along the shoreline."*

This process typically involves two main components: a **vision model** (such as a CNN or Vision Transformer) that extracts visual features from the image, and a **language model** (like an LSTM or Transformer) that uses those features to generate text. The goal is not just to recognize objects but also to describe their relationships, actions, and context in a fluent, natural sentence.

Image captioning has many practical uses, such as improving accessibility for visually impaired users, enhancing image search systems, organizing large photo collections, and supporting content generation in applications that rely on visual media.

## Picking a Model

When picking a model for image captioning, consider how complex and descriptive you want the captions to be. Most modern systems use a **vision encoder** to interpret the image and a **language decoder** to generate the text. Popular pretrained combinations include **CNN + LSTM** architectures, and more recently, **Vision Transformer (ViT)** or **CLIP** encoders paired with **Transformer-based decoders** like **GPT-2** or **T5**.  

If you want a general-purpose captioning model that works well across a wide variety of images, options like **BLIP**, **BLIP-2**, **OFA**, or **GIT** provide strong results out of the box. These models are trained on large image–text datasets and can generate fluent, contextually rich captions.  

Explore pre-trained image captioning models in the [Spark NLP Models Hub](https://sparknlp.org/models) for a variety of datasets and tasks.

<!-- #### Recommended Models for Image Captioning
- **VisionEncoderDecoder For Image Captioning:** This model can be used for generating descriptive captions based on images. It utilizes a transformer-based architecture, providing high-quality captions for various types of images. Check out the pre-trained model [`image-captioning-vit-gpt2`](https://sparknlp.org/2023/09/20/image_captioning_vit_gpt2_en.html){:target="_blank"}. -->

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageCaptioning = VisionEncoderDecoderForImageCaptioning \
    .pretrained() \
    .setDoSample(False) \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("caption")

pipeline = Pipeline().setStages([
    imageAssembler, 
    imageCaptioning
])

imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("path/to/images/folder")

model = pipeline.fit(imageDF)
result = model.transform(imageDF)

result \
    .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result") \
    .show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val imageCaptioning = VisionEncoderDecoderForImageCaptioning
  .pretrained()
  .setDoSample(false)
  .setInputCols("image_assembler")
  .setOutputCol("caption")

val pipeline = new Pipeline().setStages(Array(
  imageAssembler,
  imageCaptioning
))

val imageDF = spark.read
  .format("image")
  .option("dropInvalid", true)
  .load("path/to/images/folder")

val model = pipeline.fit(imageDF)
val result = model.transform(imageDF)

result
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")
  .show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
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
</div>

## Try Real-Time Demos!

Explore real-time image captioning outputs with our interactive demos:

- **[VisionEncoderDecoder For Image Captioning](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-VisionEncoderDecoderForImageCaptioning){:target="_blank"}**

## Useful Resources

To dive deeper into image captioning using Spark NLP, check out these useful resources:

**Notebooks**
- *[Vision Encoder-Decoder for Image Captioning](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/VisionEncoderDecoderForImageCaptioning.ipynb){:target="_blank"}*
