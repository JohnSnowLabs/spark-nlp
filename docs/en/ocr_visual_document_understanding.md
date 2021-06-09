---
layout: docs
header: true
title: Visual document understanding
permalink: /docs/en/ocr_visual_document_understanding
key: docs-ocr-visual-document-understanding
modify_date: "2020-04-08"
use_language_switcher: "Python-Scala-Java"
---

NLP models are great at processing digital text, but many real-word applications use documents with more complex formats. For example, healthcare systems often include visual lab results, sequencing reports, clinical trial forms, and other scanned documents. When we only use an NLP approach for document understanding, we lose layout and style information - which can be vital for document image understanding. New advances in multi-modal learning allow models to learn from both the text in documents (via NLP) and visual layout (via computer vision).

We provide multi-modal visual document understanding, built on Spark OCR based on the LayoutLM architecture. It achieves new state-of-the-art accuracy in several downstream tasks, including form understanding (from 70.7 to 79.3), receipt understanding (from 94.0 to 95.2) and document image classification (from 93.1 to 94.4).

Please check also webinar: [Visual Document Understanding with Multi-Modal Image & Text Mining in Spark OCR 3](https://events.johnsnowlabs.com/webinars)

## VisualDocumentClassifier

`VisualDocumentClassifier` is a DL model for classification documents using text and layout data.
Currently available pretrained model on the Tabacco3482 dataset.

#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | hocr | Сolumn name with HOCR of the document |


#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| maxSentenceLength | int | 128 | Maximum sentence length. |
| caseSensitive | boolean | false | Determines whether model is case sensitive. |
| confidenceThreshold | float | 0f| Confidence threshold. |


#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| labelCol | string | label | Name of output column with the predicted label. |
| confidenceCol | string | confidence | Name of output column with confidence. |


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

val imageToHocr = new ImageToHocr()
  .setInputCol("image")
  .setOutputCol("hocr")

val visualDocumentClassifier = VisualDocumentClassifier
  .pretrained("visual_document_classifier_tobacco3482", "en", "clinical/ocr")
  .setMaxSentenceLength(128)
  .setInputCol("hocr")
  .setLabelCol("label")
  .setConfidenceCol("conf")

val pipeline = new Pipeline()
pipeline.setStages(Array(
  imageToHocr,
  visualDocumentClassifier
))

val modelPipeline = pipeline.fit(df)

val result =  modelPipeline.transform(df)
result.select("label").show()
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

ocr = ImageToHocr() \
    .setInputCol("image") \
    .setOutputCol("hocr")

document_classifier = VisualDocumentClassifier() \
  .pretrained("visual_document_classifier_tobacco3482", "en", "clinical/ocr") \
  .setMaxSentenceLength(128) \
  .setInputCol("hocr") \
  .setLabelCol("label") \
  .setConfidenceCol("conf")

# Define pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    ocr,
    document_classifier,
    
])

result = pipeline.transform(df)
result.select("label").show()
```

</div>

Output:

```
+------+
| label|
+------+
|Letter|
+------+

```


## VisualDocumentNER

`VisualDocumentNER` is a DL model for NER documents using text and layout data.
Currently available pre-trained model on the SROIE dataset.

#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | hocr | Сolumn name with HOCR of the document |


#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| maxSentenceLength | int | 512 | Maximum sentence length. |
| caseSensitive | boolean | false | Determines whether model is case sensitive. |


#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | entities | Name of output column with entities Annotation. |


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

val imageToHocr = new ImageToHocr()
  .setInputCol("image")
  .setOutputCol("hocr")

val visualDocumentNER = VisualDocumentNER
  .pretrained("visual_document_NER_SROIE0526", "en", "public/ocr/models")
  .setMaxSentenceLength(512)
  .setInputCol("hocr")

val pipeline = new Pipeline()

pipeline.setStages(Array(
  imageToHocr,
  visualDocumentNER
))

val modelPipeline = pipeline.fit(df)
val result =  modelPipeline.transform(df)

result.select("entities").show()
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
ocr = ImageToHocr() \
    .setInputCol("image") \
    .setOutputCol("hocr")

document_ner = VisualDocumentNer() \
  .pretrained("visual_document_NER_SROIE0526", "en", "public/ocr/models") \
  .setMaxSentenceLength(512) \
  .setInputCol("hocr") \
  .setLabelCol("label") 

# Define pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    ocr,
    document_ner,    
])

result = pipeline.transform(df)
result.select("entities").show()
```

</div>

Output:

```
+-------------------------------------------------------------------------+
|entities                                                                 |
+-------------------------------------------------------------------------+
|[[entity, 0, 0, O, [word -> 0£0, token -> 0£0], []], [entity, 0, 0,      |
| B-COMPANY, [word -> AEON, token -> aeon], []], [entity, 0, 0, B-COMPANY,|
| [word -> CO., token -> co], ...                                         |
+-------------------------------------------------------------------------+
```