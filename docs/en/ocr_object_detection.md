---
layout: docs
header: true
title: Object detection
permalink: /docs/en/ocr_object_detection
key: docs-ocr-object-detection
modify_date: "2020-04-08"
use_language_switcher: "Python-Scala-Java"
---

## ImageHandwrittenDetector

`ImageHandwrittenDetector` is a DL model for detect handwritten text on the image.
It based on Cascade Region-based CNN network.

Detector support following labels:
 - 'signature'
 - 'date'
 - 'name'
 - 'title'
 - 'address'
 - 'others'


#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| scoreThreshold | float | 0.5 | Score threshold for output regions.|
| outputLabels | Array[String]| | White list for output labels.|
| labels | Array[String] | | List of labels |

#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | table_regions | array of [Coordinaties]ocr_structures#coordinate-schema)|


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

// Define transformer for detect signature
val signature_detector = ImageHandwrittenDetector
  .pretrained("image_signature_detector_gsa0628", "en", "public/ocr/models")
  .setInputCol("image")
  .setOutputCol("signature_regions")

val draw_regions = new ImageDrawRegions()
  .setInputCol("image")
  .setInputRegionsCol("signature_regions")
  .setOutputCol("image_with_regions")


pipeline = PipelineModel(stages=[
    binary_to_image,
    signature_detector,
    draw_regions
])

val data = pipeline.transform(df)

data.storeImage("image_with_regions")
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

# Define transformer for detect signature
signature_detector = ImageHandwrittenDetector \
  .pretrained("image_signature_detector_gsa0628", "en", "public/ocr/models") \
  .setInputCol("image") \
  .setOutputCol("signature_regions")

draw_regions = ImageDrawRegions() \
  .setInputCol("image") \
  .setInputRegionsCol("signature_regions") \
  .setOutputCol("image_with_regions")


pipeline = PipelineModel(stages=[
    binary_to_image,
    signature_detector,
    draw_regions
])

data = pipeline.transform(df)

display_images(data, "image_with_regions")
```

</div>

**Output:**

![image](/assets/images/ocr/signature.png)
