---
layout: docs
header: true
title: Handwriting Recognition
permalink: /docs/en/ocr_handwriting_text_recognition
key: docs-ocr-handwriting-text-recognition
modify_date: "2020-08-23"
use_language_switcher: "Python-Scala-Java"
---

## ImageHandwrittenTextRecognizer

`ImageHandwrittenTextRecognizer` is a DL model for recognizing handwritten text on the image.
It based on SPAN at https://arxiv.org/abs/2102.08742.

This model obtained the following results at line level:

{:.table-model-big}
| Dataset	| cer |	wer |
| --- | --- | --- |
| IAM	| 4.82	| 18.17 |
| RIMES	| 3.02	| 10.73 |
| READ | 2016	| 4.56	| 21.07 |


#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |


#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | handwritten_text | one text line recognized by `ImageHandwrittenTextRecognizer` |


**Example:**

<div class="tabs-box pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// Define transformer for recognizing handwritten text
val handwrittenTextRecognizer = ImageHandwrittenTextRecognizer
  .pretrained("image_handwritten_text_recognition_span0819", "en", "public/ocr/models")
  .setInputCol("image")
  .setOutputCol("handwritten_text")

pipeline = PipelineModel(stages=[
    binary_to_image,
    handwrittenTextRecognizer
])

val data = pipeline.transform(df)

data.select("handwritten_text").show(false)
```

```python
from pyspark.ml import PipelineModel
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read 
    .format("binaryFile")
    .load(imagePath)

binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image")

# Define transformer for recognizing handwritten text
recoginzer = ImageHandwrittenTextRecognizer \
  .pretrained("image_handwritten_text_recognition_span0819", "en", "public/ocr/models") \
  .setInputCol("image") \
  .setOutputCol("handwritten_text")

pipeline = PipelineModel(stages=[
    binary_to_image,
    recoginzer
])

data = pipeline.transform(df)

data.select("handwritten_text").show(truncate=False)
```

</div>

**Output:**

{:.table-model-big}
| handwritten_text |
| --- |
| a grey superimposstion of sespecteebality oven |
