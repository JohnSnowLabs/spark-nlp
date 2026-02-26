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

Image classification is a way for computers to recognize what an image contains by assigning it a single label, such as **"dog"**, **"car"**, or **"cat"**. The model looks for patterns such as shapes, colors, and textures that distinguish one class from another. It does not locate where the object is in the image or handle multiple objects; it simply identifies the overall category.

In practice, this is used to organize and tag large collections of photos (like in **Google Photos** or **stock image sites**), filter content, or power **visual search systems**. The model’s output usually includes a few possible labels with **confidence scores** that show how sure it is about each prediction.

## Picking a Model

When picking a model for image classification, think about what you are trying to achieve. For simple tasks like recognizing a few object types or when you have limited computing power, lightweight models such as **MobileNet**, **EfficientNet-Lite**, or **ResNet-18** are good starting points because they are fast and easy to deploy. If you have a larger dataset and need higher accuracy, deeper architectures like **ResNet-50**, **DenseNet**, or **EfficientNet-B7** generally perform better when properly fine-tuned.

If your images belong to a specific domain, consider using a **domain-pretrained model** that has been trained on similar data. For example, **MedNet** is designed for medical imaging, **GeoResNet** works well for satellite imagery, and **CLIP** is effective for general-purpose image and text matching. These models often outperform generic ones on domain-specific tasks.

To explore and select from a variety of models, visit [Spark NLP Models](https://sparknlp.org/models), where you can find models tailored for different tasks and datasets.

#### Recommended Models for Specific Image Classification Tasks
- **Object Detection:** For detecting objects in images, models such as [`image_classifier_vit_base_patch16_224`](https://sparknlp.org/2022/08/10/image_classifier_vit_base_patch16_224_en_3_0.html){:target="_blank"} can be used to detect objects across multiple categories.
- **Facial Expression Recognition:** Models like [`image_classifier_swin_swin_large_patch4_window12_384`](https://sparknlp.org/2023/03/23/pipeline_image_classifier_swin_swin_large_patch4_window12_384_en.html){:target="_blank"} are great for tasks that involve recognizing facial emotions.
- **Scene Classification:** To classify scenes into categories like **urban**, **rural**, or **forest**, models like [`image_classifier_vit_base_patch16_224`](https://sparknlp.org/2022/08/10/image_classifier_vit_base_patch16_224_en_3_0.html){:target="_blank"} can be applied effectively.

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

imageClassifier = ViTForImageClassification \
    .pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("class")

pipeline = Pipeline().setStages([
    imageAssembler, 
    imageClassifier
])

imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value=True) \
    .load("path/to/images/folder")

model = pipeline.fit(imageDF)
result = model.transform(imageDF)

result \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result") \
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

val imageClassifier = ViTForImageClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(
  imageAssembler,
  imageClassifier
))

val imageDF = spark.read
  .format("image")
  .option("dropInvalid", true)
  .load("path/to/images/folder")

val model = pipeline.fit(imageDF)
val result = model.transform(imageDF)

result
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result")
  .show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
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
</div>

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
