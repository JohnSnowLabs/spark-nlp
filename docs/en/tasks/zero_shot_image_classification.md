---
layout: docs
header: true
seotitle:
title: Zero-shot Image Classification
permalink: docs/en/tasks/zero_shot_image_classification
key: docs-tasks-zero-shot-image-classification
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

Zero-shot image classification is a technique that allows a model to recognize and categorize images into classes it has **never seen during training**. Instead of relying on labeled examples for every category, the model understands the relationship between images and text descriptions, enabling it to generalize to new concepts. For example, even if the model has never been trained on the class "red panda," it can still correctly identify it by understanding how the phrase "red panda" relates to its visual features.

This capability is made possible by **multimodal models** such as **CLIP** (Contrastive Language–Image Pretraining), which learn to connect images and textual descriptions through joint training on large image–text pairs. When given a new image and a set of text labels, the model compares how closely the image aligns with each label’s description and assigns the most relevant one.

## Picking a Model

When picking a model for zero-shot image classification, focus on those that are designed to understand both **images and text** in a shared representation space. The most popular and effective option is **CLIP** (Contrastive Language–Image Pretraining), developed by OpenAI. CLIP connects images with natural language descriptions, allowing it to recognize objects and concepts it was never directly trained on.  

Other strong choices include **ALIGN**, **BLIP**, and **BLIP-2**, which build on similar multimodal principles and often deliver more fluent and adaptable results. These models differ mainly in how they pair vision and text encoders—some use **Vision Transformers (ViT)** or **ResNet** for image understanding, and **Transformer-based text encoders** for processing textual labels or prompts.  

You can explore a variety of pre-trained zero-shot models on the [Spark NLP Models](https://sparknlp.org/models){:target="_blank"}, where models suited for different tasks and datasets are available.

#### Recommended Models for Zero-shot Image Classification
- **CLIP for General Zero-shot Image Classification:** Models like [`clip_vit_large_patch14 `](https://sparknlp.org/2024/09/24/clip_vit_large_patch14_en.html){:target="_blank"} and [`clip-vit-base-patch32`](https://sparknlp.org/2023/12/02/zero_shot_classifier_clip_vit_base_patch32_en.html){:target="_blank"} are well-suited for matching image content with textual labels in a zero-shot setting.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("src/test/resources/image/")

imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

candidateLabels = [
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a hen",
    "a photo of a hippo",
    "a photo of a room",
    "a photo of a tractor",
    "a photo of an ostrich",
    "a photo of an ox"
]

imageClassifier = CLIPForZeroShotClassification \
    .pretrained("clip_vit_large_patch14", "en") \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("label") \
    .setCandidateLabels(candidateLabels)

pipeline = Pipeline().setStages([imageAssembler, imageClassifier])

model = pipeline.fit(imageDF)
result = model.transform(imageDF)

result \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result") \
  .show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val imageDF = spark.read
  .format("image")
  .option("dropInvalid", true)
  .load("src/test/resources/image/")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val candidateLabels = Array(
  "a photo of a bird",
  "a photo of a cat",
  "a photo of a dog",
  "a photo of a hen",
  "a photo of a hippo",
  "a photo of a room",
  "a photo of a tractor",
  "a photo of an ostrich",
  "a photo of an ox"
)

val imageClassifier = CLIPForZeroShotClassification
  .pretrained("clip_vit_large_patch14", "en")
  .setInputCols("image_assembler")
  .setOutputCol("label")
  .setCandidateLabels(candidateLabels)

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val model = pipeline.fit(imageDF)
val result = model.transform(imageDF)

result
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result")
  .show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+-----------------+-----------------------+
|image_name       |result                 |
+-----------------+-----------------------+
|palace.JPEG      |[a photo of a room]    |
|egyptian_cat.jpeg|[a photo of a cat]     |
|hippopotamus.JPEG|[a photo of a hippo]   |
|hen.JPEG         |[a photo of a hen]     |
|ostrich.JPEG     |[a photo of an ostrich]|
|junco.JPEG       |[a photo of a bird]    |
|bluetick.jpg     |[a photo of a dog]     |
|chihuahua.jpg    |[a photo of a dog]     |
|tractor.JPEG     |[a photo of a tractor] |
|ox.JPEG          |[a photo of an ox]     |
+-----------------+-----------------------+
```
</div>

## Try Real-Time Demos!

Explore zero-shot image classification with our interactive demos:

- **[CLIP for Zero-shot Image Classification](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-CLIPForZeroShotClassification){:target="_blank"}**

## Useful Resources

**Notebooks**
- *[CLIP Classification Notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/CLIPForZeroShotClassification.ipynb){:target="_blank"}*
