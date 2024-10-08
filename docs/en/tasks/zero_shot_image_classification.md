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

**Zero-shot image classification** is a technique in computer vision where a model can classify images into categories that it has never seen before during training. This is achieved by leveraging semantic relationships between the image data and textual descriptions of classes, enabling models to predict labels without specific training on each category.

This task is particularly useful for scenarios where obtaining labeled data for every possible category is challenging or expensive, such as real-world applications in e-commerce, media, or biology. Zero-shot classification can help scale image recognition systems without constantly retraining them for new categories.

## How Zero-shot Image Classification Works

The key idea behind zero-shot learning is the generalization capability of models. Instead of being restricted to the labels encountered during training, the model uses external knowledge—typically in the form of text or word embeddings—to make predictions about new classes.

In Spark NLP, zero-shot image classification leverages models like CLIP (Contrastive Language–Image Pretraining), which are trained to understand both visual and textual data. These models align the visual representations of images with the semantic representations of text, allowing them to match unseen image categories based on their descriptions.

Some common use cases include:

- **Classifying new product images** in an e-commerce platform without retraining the model for every new product.
- **Detecting rare or new species of animals** using images in wildlife research.
- **Media categorization** for content recommendation engines where new labels continuously emerge.

## Picking a Model

When choosing a model for zero-shot image classification, you need to consider several factors:

- **Text and Image Alignment:** Choose models that are good at matching visual features to text-based descriptions.
- **Task Complexity:** Depending on the complexity of the task, a larger pre-trained model like CLIP or a fine-tuned ViT model might perform better.
- **Efficiency:** While zero-shot classification saves time by avoiding retraining, some models are more resource-intensive than others. Make sure the model is efficient enough for your computational setup.

You can explore a variety of pre-trained zero-shot models on the [Spark NLP Models](https://sparknlp.org/models){:target="_blank"}, where models suited for different tasks and datasets are available.

#### Recommended Models for Zero-shot Image Classification
- **CLIP for General Zero-shot Image Classification:** Models like [`clip_vit_large_patch14 `](https://sparknlp.org/2024/09/24/clip_vit_large_patch14_en.html){:target="_blank"} and [`clip-vit-base-patch32`](https://sparknlp.org/2023/12/02/zero_shot_classifier_clip_vit_base_patch32_en.html){:target="_blank"} are well-suited for matching image content with textual labels in a zero-shot setting.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Loading images into a Spark DataFrame, with an option to discard invalid images
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("src/test/resources/image/")

# Assembling image data using the ImageAssembler, preparing the input images for further processing
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

# Defining candidate labels for zero-shot classification
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

# Initializing the CLIPForZeroShotClassification model
imageClassifier = CLIPForZeroShotClassification \
    .pretrained("clip_vit_large_patch14", "en") \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("label") \
    .setCandidateLabels(candidateLabels)

# Defining a Spark ML pipeline with two stages: the ImageAssembler and the CLIP image classifier
pipeline = Pipeline().setStages([imageAssembler, imageClassifier])

# Fitting the pipeline on the image DataFrame and transforming the data to apply classification
pipelineDF = pipeline.fit(imageDF).transform(imageDF)

# Selecting the image file name and the predicted label result, displaying the output in a readable format
pipelineDF \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result") \
  .show(truncate=False)

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
```scala
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// Loading image data into a Spark DataFrame, removing any invalid images
val imageDF = ResourceHelper.spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

// Assembling the images with the ImageAssembler, which prepares image data for processing
val imageAssembler: ImageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

// Defining an array of candidate labels for zero-shot image classification
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

// Initializing the CLIPForZeroShotClassification model, setting input and output columns
// The model classifies images based on comparison to the candidate labels
val imageClassifier = CLIPForZeroShotClassification
  .pretrained()  // Loading a pretrained CLIP model 
  .setInputCols("image_assembler")
  .setOutputCol("label")
  .setCandidateLabels(candidateLabels)

// Creating and running the Spark ML pipeline with the image assembler and classifier
val pipeline =
  new Pipeline().setStages(Array(imageAssembler, imageClassifier)).fit(imageDF).transform(imageDF)

// Selecting and displaying the image file name and classification result
pipeline
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result")  // Extracting image names and their classification labels
  .show(truncate = false)

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

Learn zero-shot image classification with Spark NLP:

**Notebooks**
- *[CLIP Classification Notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/CLIPForZeroShotClassification.ipynb){:target="_blank"}*

Discover how to classify images without labeled data.
