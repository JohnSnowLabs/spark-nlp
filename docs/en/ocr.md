---
layout: docs
header: true
seotitle: Visual NLP | John Snow Labs
title: Visual NLP (Spark OCR)
permalink: /docs/en/ocr
key: docs-ocr
modify_date: "2020-04-08"
use_language_switcher: "Python-Scala-Java"
show_nav: true
sidebar:
    nav: spark-ocr
---
Spark OCR is another commercial extension of Spark NLP for optical character recognition from images, scanned PDF documents, Microsoft DOCX and DICOM files. If you want to try it out on your own documents click on the below button:

{:.btn-block}
[Try Free](https://www.johnsnowlabs.com/spark-nlp-try-free/){:.button.button--primary.button--rounded.button--lg}


Spark OCR is built on top of ```Apache Spark``` and offers the following capabilities:
  - Image pre-processing algorithms to improve text recognition results:
  - Adaptive thresholding & denoising
  - Skew detection & correction
  - Adaptive scaling
  - Layout Analysis & region detection
  - Image cropping
  - Removing background objects
- Text recognition, by combining NLP and OCR pipelines:
  - Extracting text from images (optical character recognition)
  - Support English, German, French, Spanish, Russian, Vietnamese and Arabic languages
  - Extracting data from tables
  - Recognizing and highlighting named entities in PDF documents
  - Masking sensitive text in order to de-identify images
- Table detection and recognition from images
- Signature detection
- Visual document understanding
  - Document classification
  - Visual NER
- Output generation in different formats:
  - PDF, images, or DICOM files with annotated or masked entities
  - Digital text for downstream processing in Spark NLP or other libraries
  - Structured data formats (JSON and CSV), as files or Spark data frames
- Scale out: distribute the OCR jobs across multiple nodes in a Spark cluster.
- Frictionless unification of OCR, NLP, ML & DL pipelines.


## Spark OCR Workshop

If you prefer learning by example, click the button below to checkout the workshop repository full of fresh examples.

[Spark OCR Workshop](https://github.com/JohnSnowLabs/spark-ocr-workshop){:.button.button--primary.button--rounded.button--md}

Below, you can follow a more theoretical and thorough quick start guide.

## Quickstart Examples

<div class="h3-box" markdown="1">

### Images

The following code example creates an OCR Pipeline for processing image(s). The image file(s) can contain complex layout like columns, tables, images inside.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers._

val imagePath = "path to image files"

// Read image files as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)

// Transform binary content to image
val binaryToImage = new BinaryToImage()
  .setInputCol("content")
  .setOutputCol("image")

// OCR
val ocr = new ImageToText()
  .setInputCol("image")
  .setOutputCol("text")

// Define Pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  binaryToImage,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = modelPipeline.transform(df)

data.show()
```

```python
from pyspark.ml import PipelineModel

from sparkocr.transformers import *

imagePath = "path to image files"

# Read image files as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath)

# Transform binary content to image
binaryToImage = BinaryToImage() \
  .setInputCol("content") \
  .setOutputCol("image")

# OCR
ocr = ImageToText() \
  .setInputCol("image") \
  .setOutputCol("text")

# Define Pipeline
pipeline = PipelineModel(stages=[
  binaryToImage,
  ocr
])

data = pipeline.transform(df)

data.show()
```

</div></div><div class="h3-box" markdown="1">

### Scanned PDF files

Next sample provides an example of OCR Pipeline for processing PDF files containing image data. In this case, the [PdfToImage](ocr_pipeline_components#pdftoimage) transformer is used to convert PDF file to a set of images.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers._

val imagePath = "path to pdf files"

// Read pdf files as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)

// Transform PDF file to the image
val pdfToImage = new PdfToImage()
  .setInputCol("content")
  .setOutputCol("image")

// OCR
val ocr = new ImageToText()
  .setInputCol("image")
  .setOutputCol("text")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  pdfToImage,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = modelPipeline.transform(df)

data.show()
```

```python
from pyspark.ml import PipelineModel

from sparkocr.transformers import *

imagePath = "path to pdf files"

# Read pdf files as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath)

# Transform PDF file to the image
pdfToImage = PdfToImage() \
  .setInputCol("content") \
  .setOutputCol("image")

# OCR
ocr = ImageToText() \
  .setInputCol("image") \
  .setOutputCol("text")

# Define pipeline
pipeline = PipelineModel(stages=[
  pdfToImage,
  ocr
])

data = pipeline.transform(df)

data.show()
```

</div></div><div class="h3-box" markdown="1">

### PDF files (scanned or text) 

In the following code example we will create OCR Pipeline for processing PDF files that contain text or image data.

For each PDF file, this pipeline will:
 * extract the text from document and save it to the `text` column
 * if `text` contains less than 10 characters (so the document isn't PDF with text layout) it will process the PDF file as a scanned document:
    - convert PDF file to an image
    - detect and split image to regions
    - run OCR and save output to the `text` column


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers._

val imagePath = "path to PDF files"

// Read PDF files as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)

// Extract text from PDF text layout
val pdfToText = new PdfToText()
  .setInputCol("content")
  .setOutputCol("text")
  .setSplitPage(false)

// In case of `text` column contains less then 10 characters,
// pipeline run PdfToImage as fallback method
val pdfToImage = new PdfToImage()
  .setInputCol("content")
  .setOutputCol("image")
  .setFallBackCol("text")
  .setMinSizeBeforeFallback(10)

// OCR
val ocr = new ImageToText()
  .setInputCol("image")
  .setOutputCol("text")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  pdfToText,
  pdfToImage,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = modelPipeline.transform(df)

data.show()
```

```python
from pyspark.ml import PipelineModel

from sparkocr.transformers import *


imagePath = "path to PDF files"

# Read PDF files as binary file
df = spark.read
  .format("binaryFile")
  .load(imagePath)

# Extract text from PDF text layout
pdfToText = PdfToText() \
  .setInputCol("content") \
  .setOutputCol("text") \
  .setSplitPage(false)

# In case of `text` column contains less then 10 characters,
# pipeline run PdfToImage as fallback method
pdfToImage = PdfToImage() \
  .setInputCol("content") \
  .setOutputCol("image") \
  .setFallBackCol("text") \
  .setMinSizeBeforeFallback(10)

# OCR
ocr = ImageToText() \
  .setInputCol("image") \
  .setOutputCol("text")

# Define pipeline
pipeline = PipelineModel(stages=[
  pdfToText,
  pdfToImage,
  ocr,
])

data = pipeline.transform(df)

data.show()
```

</div></div><div class="h3-box" markdown="1">

### Images (streaming mode)

Next code segments provide an example of streaming OCR pipeline. It processes images and stores results to memory table.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
val imagePath = "path folder with images"

val batchDataFrame = spark.read.format("binaryFile").load(imagePath).limit(1)
    
val pipeline = new Pipeline()
pipeline.setStages(Array(
  binaryToImage,
  binarizer,
  ocr
))

val modelPipeline = pipeline.fit(batchDataFrame)

// Read files in streaming mode
val dataFrame = spark.readStream
  .format("binaryFile")
  .schema(batchDataFrame.schema)
  .load(imagePath)

// Call pipeline and store results to 'results' memory table
val query = modelPipeline.transform(dataFrame)
  .select("text", "exception")
  .writeStream
  .format("memory")
  .queryName("results")
  .start()
```

```python
imagePath = "path folder with images"

batchDataFrame = spark.read.format("binaryFile").load(imagePath).limit(1)
    
pipeline = Pipeline()
pipeline.setStages(Array(
  binaryToImage,
  binarizer,
  ocr
))

modelPipeline = pipeline.fit(batchDataFrame)

# Read files in streaming mode
dataFrame = spark.readStream
  .format("binaryFile")
  .schema(batchDataFrame.schema)
  .load(imagePath)

# Call pipeline and store results to 'results' memory table
query = modelPipeline.transform(dataFrame) \
  .select("text", "exception") \
  .writeStream() \
  .format("memory") \
  .queryName("results") \
  .start()
```

</div>

For getting results from memory table following code could be used:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
spark.table("results").select("path", "text").show()
```

```python
spark.table("results").select("path", "text").show()
```

</div>

More details about Spark Structured Streaming could be found in [spark documentation](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).
{:.info}

</div><div class="h3-box" markdown="1">

## Advanced Topics

### Error Handling

Pipeline execution would not be interrupted in case of the runtime exceptions 
while processing some records. 

In this case OCR transformers would fill _exception_ column that contains _transformer name_ and _exception_.

**NOTE:** Storing runtime errors to the _exception_ field allows to process batch of files. 
{:.info}

</div><div class="h3-box" markdown="1">

#### Output

Here is an output with exception when try to process js file using OCR pipeline:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
result.select("path", "text", "exception").show(2, false)
```

```python
result.select("path", "text", "exception").show(2, False)
```

</div>

```
+----------------------+-------------------------------------------+-----------------------------------------------------+
|path                  |text                                       |exception                                            |
+----------------------+-------------------------------------------+-----------------------------------------------------+
|file:jquery-1.12.3.js |                                           |BinaryToImage_c0311dc62161: Can't open file as image.|
|file:image.png        |I prefer the morning flight through Denver |null                                                 |
+----------------------+-------------------------------------------+-----------------------------------------------------+
```

</div>

### Performance

In case of big count of text PDF's in dataset
need have manual partitioning for avoid skew in partitions and effective utilize resources. 
For example the randomization could be used.
