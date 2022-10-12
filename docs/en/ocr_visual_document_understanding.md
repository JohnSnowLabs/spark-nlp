---
layout: docs
header: true
seotitle: Spark OCR | John Snow Labs
title: Visual document understanding
permalink: /docs/en/ocr_visual_document_understanding
key: docs-ocr-visual-document-understanding
modify_date: "2020-04-08"
use_language_switcher: "Python-Scala-Java"
show_nav: true
sidebar:
    nav: spark-ocr
---

NLP models are great at processing digital text, but many real-word applications use documents with more complex formats. For example, healthcare systems often include visual lab results, sequencing reports, clinical trial forms, and other scanned documents. When we only use an NLP approach for document understanding, we lose layout and style information - which can be vital for document image understanding. New advances in multi-modal learning allow models to learn from both the text in documents (via NLP) and visual layout (via computer vision).

We provide multi-modal visual document understanding, built on Spark OCR based on the LayoutLM architecture. It achieves new state-of-the-art accuracy in several downstream tasks, including form understanding (from 70.7 to 79.3), receipt understanding (from 94.0 to 95.2) and document image classification (from 93.1 to 94.4).

Please check also webinar: [Visual Document Understanding with Multi-Modal Image & Text Mining in Spark OCR 3](https://events.johnsnowlabs.com/webinars)

## VisualDocumentClassifier

`VisualDocumentClassifier` is a DL model for document classification using text and layout data.
Currently available pretrained model on the Tobacco3482 dataset, that contains 3482 images belonging 
to 10 different classes (Resume, News, Note, Advertisement, Scientific, Report, Form, Letter, Email and Memo)

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


<div class="tabs-box tabs-new pt0" markdown="1">

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
Currently available pre-trained model on the SROIE dataset. The dataset has 1000 whole 
scanned receipt images.

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
| whiteList | Array[String] | | Whitelist of output labels |

#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | entities | Name of output column with entities Annotation. |


**Example:**


<div class="tabs-box tabs-new pt0" markdown="1">

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

## VisualDocumentNERv2

`VisualDocumentNERv2` is a DL model for NER documents which is an improved version of `VisualDocumentNER`. There is available pretrained model trained on FUNSD dataset.
The dataset comprises 199 real, fully annotated, scanned forms.

#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCols | Array[String] |  | Сolumn names for tokens of the document and image|


#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| maxSentenceLength | int | 512 | Maximum sentence length. |
| whiteList | Array[String] | | Whitelist of output labels |

#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | entities | Name of output column with entities Annotation. |


**Example:**


<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

var dataFrame = spark.read.format("binaryFile").load(imagePath)

var bin2imTransformer = new BinaryToImage()
bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

val ocr = new ImageToHocr()
  .setInputCol("image")
  .setOutputCol("hocr")
  .setIgnoreResolution(false)
  .setOcrParams(Array("preserve_interword_spaces=0"))

val tokenizer = new HocrTokenizer()
  .setInputCol("hocr")
  .setOutputCol("token")

val visualDocumentNER = VisualDocumentNERv2
  .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")
  .setInputCols(Array("token", "image"))

val pipeline = new Pipeline()
  .setStages(Array(
    bin2imTransformer,
    ocr,
    tokenizer,
    visualDocumentNER
  ))

val results = pipeline
  .fit(dataFrame)
  .transform(dataFrame)
  .select("entities")
  .cache()

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

binToImage = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image")

ocr = ImageToHocr()\
    .setInputCol("image")\
    .setOutputCol("hocr")\
    .setIgnoreResolution(False)\
    .setOcrParams(["preserve_interword_spaces=0"])

tokenizer = HocrTokenizer()\
    .setInputCol("hocr")\
    .setOutputCol("token")

ner = VisualDocumentNerV2()\
    .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")\
    .setInputCols(["token", "image"])\
    .setOutputCol("entities")

pipeline = PipelineModel(stages=[
    binToImage,
    ocr,
    tokenizer,
    ner
    ])

result = pipeline.transform(df)
result.withColumn('filename', path\_array.getItem(f.size(path_array)- 1)) \
    .withColumn("exploded_entities", f.explode("entities")) \
    .select("filename", "exploded_entities") \
    .show(truncate=False)
```

</div>

Output sample:

```
+---------+-------------------------------------------------------------------------------------------------------------------------+
|filename |exploded_entities                                                                                                        |
+---------+-------------------------------------------------------------------------------------------------------------------------+
|form1.jpg|[entity, 0, 6, i-answer, [x -> 1027, y -> 89, height -> 19, confidence -> 96, word -> Version:, width -> 90], []]        |
|form1.jpg|[entity, 25, 35, b-header, [x -> 407, y -> 190, height -> 37, confidence -> 96, word -> Institution, width -> 241], []]  |
|form1.jpg|[entity, 37, 40, i-header, [x -> 667, y -> 190, height -> 37, confidence -> 96, word -> Name, width -> 130], []]         |
|form1.jpg|[entity, 42, 52, b-question, [x -> 498, y -> 276, height -> 19, confidence -> 96, word -> Institution, width -> 113], []]|
|form1.jpg|[entity, 54, 60, i-question, [x -> 618, y -> 276, height -> 19, confidence -> 96, word -> Address, width -> 89], []]     |
+---------+-------------------------------------------------------------------------------------------------------------------------+
```

## FormRelationExtractor

`FormRelationExtractor` detect relation between keys and values detected by `VisualDocumentNERv2`.

It can detect relations only for key/value in same line.

#### Input Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | String |  | Column name for entities Annotation|


#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| lineTolerance | int | 15 | Line tolerance in pixels. This is the space between lines that will be assumed. It is used for grouping text regions by lines. |
| keyPattern | String | question | Pattern of entity name for keys in form. |
| valuePattern | String | answer | Pattern of entity name for values in form. |

#### Output Columns

{:.table-model-big}
| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | relations | Name of output column with relation Annotations. |


**Example:**


<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

var dataFrame = spark.read.format("binaryFile").load(imagePath)

var bin2imTransformer = new BinaryToImage()
bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

val ocr = new ImageToHocr()
  .setInputCol("image")
  .setOutputCol("hocr")
  .setIgnoreResolution(false)
  .setOcrParams(Array("preserve_interword_spaces=0"))

val tokenizer = new HocrTokenizer()
  .setInputCol("hocr")
  .setOutputCol("token")

val visualDocumentNER = VisualDocumentNERv2
  .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")
  .setInputCols(Array("token", "image"))

val relExtractor = new FormRelationExtractor()
  .setInputCol("entities")
  .setOutputCol("relations")

val pipeline = new Pipeline()
  .setStages(Array(
    bin2imTransformer,
    ocr,
    tokenizer,
    visualDocumentNER,
    relExtractor
  ))

val results = pipeline
  .fit(dataFrame)
  .transform(dataFrame)
  .select("relations")
  .cache()

results.select(explode("relations")).show(3, False)
```

```python
from pyspark.ml import PipelineModel
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read 
    .format("binaryFile")
    .load(imagePath)

binToImage = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image")

ocr = ImageToHocr()\
    .setInputCol("image")\
    .setOutputCol("hocr")\
    .setIgnoreResolution(False)\
    .setOcrParams(["preserve_interword_spaces=0"])

tokenizer = HocrTokenizer()\
    .setInputCol("hocr")\
    .setOutputCol("token")

ner = VisualDocumentNerV2()\
    .pretrained("layoutlmv2_funsd", "en", "clinical/ocr")\
    .setInputCols(["token", "image"])\
    .setOutputCol("entities")

rel_extractor = FormRelationExtractor() \
    .setInputCol("entities") \
    .setOutputCol("relations")

pipeline = PipelineModel(stages=[
    binToImage,
    ocr,
    tokenizer,
    ner,
    rel_extractor
    ])

result = pipeline.transform(df)
result.select(explode("relations")).show(3, False)
```

</div>

Output sample:

```
+---------------------------------------------------------------------+
|col                                                                  |
+---------------------------------------------------------------------+
|[relation, 112, 134, Name: Dribbler, bbb, [bbox1 -> 58 478 69 19, ...|
|[relation, 136, 161, Study Date: 12-09-2006, 6:34, [bbox1 -> 431 ... |
|[relation, 345, 361, BP: 120 80 mmHg, [bbox1 -> 790 478 30 19, ...   |
+---------------------------------------------------------------------+
```
