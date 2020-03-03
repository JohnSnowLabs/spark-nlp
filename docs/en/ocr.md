---
layout: article
title: Spark OCR (Licensed)
permalink: /docs/en/ocr
key: docs-ocr
modify_date: "2020-02-20"
use_language_switchter: "Python-Scala-Java"
---
Spark OCR provides set of Spark ML transformers/estimators that help users create and use OCR pipelines.
It built on top of Apache Spark and Tesseract OCR.

# OCR Pipelines

Using Spark OCR it possible to build pipelines for recognition text from:
 - scanned image(s) (png, tiff, jpeg ...)
 - selectable PDF (that contains text layout)
 - not selectable PDF (that contains scanned text as an image)
 
It contains set of tools for

 - PDF processing transformers which extract text and images from PDF files
 - Image pre-processing (scaling, binarization, skew correction, etc.) transformers
 - Splitting image to regions analyzers and transformers
 - Characters recognition using TesseractOCR estimator

More details on transformers/estimators could be found in further section [OCR Pipeline Components](#ocr-pipeline-components)

## Requirements

Spark OCR is built on top of **Apache Spark 2.4.4**. This is the **only** supported release.

It is recommended to have basic knowledge of the framework and a working environment before using Spark OCR. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.4/index.html) to get started with Spark.

Spark OCR required Tesseract 4.1.+.

## Installation

### Installing Tesseract

As mentioned above, if you are dealing with scanned images instead of test-selectable PDF files you need to install `tesseract 4.x+` on all the nodes in your cluster. Here how you can install it on Ubuntu/Debian:

```bash
apt-get install tesseract-ocr
```

In `Databricks` this command may result in installing `tesseract 3.x` instead of version `4.x`.

You can simply run this `init script` to install `tesseract 4.x` in your Databricks cluster:

```bash
#!/bin/bash
sudo apt-get install -y g++ # or clang++ (presumably)
sudo apt-get install -y autoconf automake libtool
sudo apt-get install -y pkg-config
sudo apt-get install -y libpng-dev
sudo apt-get install -y libjpeg8-dev
sudo apt-get install -y libtiff5-dev
sudo apt-get install -y zlib1g-dev
​
wget http://www.leptonica.org/source/leptonica-1.74.4.tar.gz
tar xvf leptonica-1.74.4.tar.gz
cd leptonica-1.74.4
./configure
make
sudo make install
​
git clone --single-branch --branch 4.1 https://github.com/tesseract-ocr/tesseract.git
cd tesseract
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
​
tesseract -v
```

Mac OS:

```bash
brew install tesseract
```

### Install Python package

Install python package using pip:

```bash
pip install spark-ocr==1.1.0 --extra-index-url #### --ignore-installed
```

The #### is a secret url only avaliable for users with license, if you
have not received it please contact us at info@johnsnowlabs.com.

### Spark OCR from Scala

You can start a spark REPL with Scala by running in your terminal a
spark-shell including the com.johnsnowlabs.nlp:spark-ocr_2.11:1.0.0 package:

```bash
spark-shell --jars ####
```

The #### is a secret url only avaliable for users with license, if you
have not received it please contact us at info@johnsnowlabs.com.

### Start Spark OCR Session from Python and Scala

The following will initialize the spark session in case you have run
the jupyter notebook directly. If you have started the notebook using
pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1
minute) as the jar from the server needs to be loaded.

The #### in .config("spark.jars", "####") is a secret code, if you have
not received it please contact us at info@johnsnowlabs.com.

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession
    .builder()
    .appName("Spark OCR")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.driver.maxResultSize", "2G")
    .config("spark.jars", "####")
    .getOrCreate()
```

```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Spark OCR") \
    .master("local[*]") \
    .config("spark.driver.memory", "4G") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars", "####") \
    .getOrCreate()
```

Another way to initialoze SparkSession with Spark OCR to use `start` function in Python:

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparkocr import start
   
spark = start(secret=secret)
```

```scala
// Implemented only for Python


```

# Spark OCR Workshop

If you prefer learning by example, check this repository:

[Spark OCR Workshop](https://github.com/JohnSnowLabs/spark-ocr-workshop){:.button.button--primary.button--rounded.button--md}

It is full of fresh examples.

Below, you can follow into a more theoretical and thorough quick start guide.


# Quickstart Examples

## Images

In the following code example we will create OCR Pipeline for processing image(s). 
The image file(s) can contain complex layout like columns, tables, images inside.

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
val ocr = new TesseractOcr()
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
ocr = TesseractOcr() \
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

## Scanned PDF files

Next sample provides an example of OCR Pipeline for processing PDF files with image data.
In this case it needed to use [PdfToImage](#pdftoimage) transformer to convert PDF file to the set of images.


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
val ocr = new TesseractOcr()
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
ocr = TesseractOcr() \
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

## PDF files (scanned or text) 

In the following code example we will create OCR Pipeline for processing 
PDF files containing text or image data.

While running pipeline for each PDF file, it will:
 * extract the text from document and save it to the `text` column
 * if `text` contains less than 10 characters (so the document isn't PDF with text layout), 
 process PDF file as an scanned document:
    - convert PDF file to an image
    - detect and split image to regions
    - run OCR and save output to the `text` column


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
val ocr = new TesseractOcr()
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
ocr = TesseractOcr() \
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

## Images (streaming mode)

Next code segments provide an example of streaming OCR pipeline.
It processes images and stores results to memory table.

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

For getting results from memory table following code could be used:

{% include programmingLanguageSelectScalaPython.html %}

```scala
spark.table("results").select("path", "text").show()
```

```python
spark.table("results").select("path", "text").show()
```

More details about Spark Structured Streaming could be found in 
[spark documentation](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).
{:.info}

# Pipeline components

## PDF processing

Next section describes the transformers for deal with PDF files: extracting text and image data from PDF
files.

### PdfToText

`PDFToText` extracts text from selectable PDF (with text layout).

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | text | binary representation of the PDF document |
| originCol | string | path | path to the original file |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| splitPage | bool | true | whether it needed to split document to pages |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | text | extracted text |
| pageNumCol | string | pagenum | page number or 0 when `splitPage = false` |


NOTE: For setting parameters use `setParamName` method.
{:.info}

**Example**


{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.PdfToText

val pdfPath = "path to pdf with text layout"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val transformer = new PdfToText()
  .setInputCol("content")
  .setOutputCol("text")
  .setPageNumCol("pagenum")
  .setSplitPage(true)

val data = transformer.transform(df)

data.select("pagenum", "text").show()
```

```python
from sparkocr.transformers import *

pdfPath = "path to pdf with text layout"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

transformer = PdfToText() \
  .setInputCol("content") \
  .setOutputCol("text") \
  .setPageNumCol("pagenum") \
  .setSplitPage(true)

data = transformer.transform(df)

data.select("pagenum", "text").show()
```


**Output:**

```
+-------+----------------------+
|pagenum|text                  |
+-------+----------------------+
|0      |This is a page.       |
|1      |This is another page. |
|2      |Yet another page.     |
+-------+----------------------+
```

### PdfToImage

`PdfToImage` renders PDF to an image. To be used with scanned PDF documents.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the PDF document |
| originCol | string | path | path to the original file |
| fallBackCol | string | text | extracted text from previous method for detect if need to run transformer as fallBack |


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| splitPage | bool | true | whether it needed to split document to pages |
| minSizeBeforeFallback | int | 10 | minimal count of characters to extract to decide, that the document is the PDF with text layout |
| imageType | [ImageType](#imagetype) | `ImageType.TYPE_BYTE_GRAY` | type of the image |
| resolution | int | 300 | Output image resolution in dpi |
| keepInput | boolean | false | Keep input column in dataframe. By default it is dropping. |
| partitionNum | int | 0 | Number of partitions (0 value - without repartition) |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image | extracted image struct ([Image schema](#image-schema)) |
| pageNumCol | string | pagenum | page number or 0 when `splitPage = false` |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.PdfToImage

val pdfPath = "path to pdf"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToImage = new PdfToImage()
 .setInputCol("content")
 .setOutputCol("text")
 .setPageNumCol("pagenum")
 .setSplitPage(true)

val data =  pdfToImage.transform(df)

data.select("pagenum", "text").show()
```

```python
from sparkocr.transformers import *

pdfPath = "path to pdf"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

pdfToImage = PdfToImage() \
 .setInputCol("content") \
 .setOutputCol("text") \
 .setPageNumCol("pagenum") \
 .setSplitPage(true)

data =  pdfToImage.transform(df)

data.select("pagenum", "text").show()
```

### ImageToPdf

`ImageToPdf` transform image to Pdf document.
If dataframe contains few records for same origin path, it groups image by origin
column and create multipage PDF document.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema))  |
| originCol | string | path | path to the original file |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | content | binary representation of the PDF document |


**Example:**

Read images and store it as single page PDF documents.


{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read.format("binaryFile").load(imagePath)

// Define transformer for convert to Image struct
val binaryToImage = new BinaryToImage()
  .setInputCol("content")
  .setOutputCol("image")

// Define transformer for store to PDF
val imageToPdf = new ImageToPdf()
  .setInputCol("image")
  .setOutputCol("content")

// Call transformers
val image_df = binaryToImage.transform(df)
val pdf_df =  pdfToImage.transform(image_df)

pdf_df.select("content").show()
```

```python
from sparkocr.transformers import *

pdfPath = "path to pdf"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

# Define transformer for convert to Image struct
binaryToImage = BinaryToImage() \
  .setInputCol("content") \
  .setOutputCol("image")

# Define transformer for store to PDF
imageToPdf = ImageToPdf() \
  .setInputCol("image") \
  .setOutputCol("content")

# Call transformers
image_df = binaryToImage.transform(df)
pdf_df =  pdfToImage.transform(image_df)

pdf_df.select("content").show()
```

### TextToPdf

`TextToPdf` renders ocr results to PDF document as text layout. Each symbol will render to same position
with same font size as in original image or PDF.
If dataframe contains few records for same origin path, it groups image by origin
column and create multipage PDF document.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | positions | column with positions struct  |
| inputImage | string | image | image struct ([Image schema](#image-schema))  |
| inputText | string | text | column name with recognized text |
| originCol | string | path | path to the original file |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | pdf | binary representation of the PDF document |


**Example:**

Read PDF document, run OCR and render results to PDF document.


{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.ocr.transformers._

val pdfPath = "path to pdf"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToImage = new PdfToImage()
  .setInputCol("content")
  .setOutputCol("image_raw")
  .setResolution(400)

val binarizer = new ImageBinarizer()
  .setInputCol("image_raw")
  .setOutputCol("image")
  .setThreshold(130)

val ocr = new TesseractOcr()
  .setInputCol("image")
  .setOutputCol("text")
  .setIgnoreResolution(false)
  .setPageSegMode(PageSegmentationMode.SPARSE_TEXT)
  .setConfidenceThreshold(60)

val textToPdf = new TextToPdf()
  .setInputCol("positions")
  .setInputImage("image")
  .setOutputCol("pdf")

val pipeline = new Pipeline()
pipeline.setStages(Array(
 pdfToImage,
 binarizer,
 ocr,
 textToPdf
))

val modelPipeline = pipeline.fit(df)

val pdf = modelPipeline.transform(df)

val pdfContent = pdf.select("pdf").collect().head.getAs[Array[Byte]](0)

// store to file
val tmpFile = Files.createTempFile(suffix=".pdf").toAbsolutePath.toString
val fos = new FileOutputStream(tmpFile)
fos.write(pdfContent)
fos.close()
println(tmpFile)
```

```python
from sparkocr.transformers import *

pdfPath = "path to pdf"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

pdf_to_image = PdfToImage() \
    .setInputCol("content") \
    .setOutputCol("image_raw")

binarizer = ImageBinarizer() \
    .setInputCol("image_raw") \
    .setOutputCol("image") \
    .setThreshold(130)

ocr = TesseractOcr() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
    .setConfidenceThreshold(60)

textToPdf = TextToPdf() \
    .setInputCol("positions") \
    .setInputImage("image") \
    .setOutputCol("pdf")

pipeline = PipelineModel(stages=[
    pdf_to_image,
    binarizer,
    ocr,
    textToPdf
])

result = pipeline.transform(df).collect()

# Store to file for debug
with open("test.pdf", "wb") as file:
    file.write(result[0].pdf)










```

### PdfDrawRegions

`PdfDrawRegions` transformer for drawing regions to Pdf document.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the PDF document |
| originCol | string | path | path to the original file |
| inputRegionsCol | string | region | input column which contain regions |


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| lineWidth | integer | 1 | line width for draw regions |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | pdf_regions | binary representation of the PDF document |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import java.io.FileOutputStream
import java.nio.file.Files

import com.johnsnowlabs.ocr.transformers._
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.util.io.ReadAs

val pdfPath = "path to pdf"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToText = new PdfToText()
  .setInputCol("content")
  .setOutputCol("text")
  .setSplitPage(false)

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val entityExtractor = new TextMatcher()
  .setInputCols("sentence", "token")
  .setEntities("test-chunks.txt", ReadAs.TEXT)
  .setOutputCol("entity")

val positionFinder = new PositionFinder()
  .setInputCols("entity")
  .setOutputCol("coordinates")
  .setPageMatrixCol("positions")
  .setMatchingWindow(10)
  .setPadding(2)

val pdfDrawRegions = new PdfDrawRegions()
  .setInputRegionsCol("coordinates")

// Create pipeline
val pipeline = new Pipeline()
  .setStages(Array(
    pdfToText,
    documentAssembler,
    sentenceDetector,
    tokenizer,
    entityExtractor,
    positionFinder,
    pdfDrawRegions
  ))

val pdfWithRegions = pipeline.fit(df).transform(df)

val pdfContent = pdfWithRegions.select("pdf_regions").collect().head.getAs[Array[Byte]](0)

// store to pdf to tmp file
val tmpFile = Files.createTempFile("with_regions_", s".pdf").toAbsolutePath.toString
val fos = new FileOutputStream(tmpFile)
fos.write(pdfContent)
fos.close()
println(tmpFile)
```

```python
from pyspark.ml import Pipeline

from sparkocr.transformers import *
from sparknlp.annotator import *
from sparknlp.base import *

pdfPath = "path to pdf"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

pdf_to_text = PdfToText() \
    .setInputCol("content") \
    .setOutputCol("text") \
    .setPageNumCol("page") \
    .setSplitPage(False)

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

entity_extractor = TextMatcher() \
    .setInputCols("sentence", "token") \
    .setEntities("./sparkocr/resources/test-chunks.txt", ReadAs.TEXT) \
    .setOutputCol("entity")

position_finder = PositionFinder() \
    .setInputCols("entity") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(10) \
    .setPadding(2)

draw = PdfDrawRegions() \
    .setInputRegionsCol("coordinates") \
    .setOutputCol("pdf_with_regions") \
    .setInputCol("content") \
    .setLineWidth(1)

pipeline = Pipeline(stages=[
    pdf_to_text,
    document_assembler,
    sentence_detector,
    tokenizer,
    entity_extractor,
    position_finder,
    draw
])

pdfWithRegions = pipeline.fit(df).transform(df)

pdfContent = pdfWithRegions.select("pdf_regions").collect().head.getAs[Array[Byte]](0)

# store to pdf to tmp file
with open("test.pdf", "wb") as file:
    file.write(pdfContent[0].pdf_regions)  



```

Results:

![Result with regions](/assets/images/ocr/with_regions.png)

## Image pre-processing

Next section describes the transformers for image pre-processing: scaling, binarization, skew correction, etc.

### BinaryToImage

`BinaryToImage` transforms image (loaded as binary file) to image struct.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the image |
| originCol | string | path | path to the original file |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image | extracted image struct ([Image schema](#image-schema)) |

**Scala example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.BinaryToImage

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read.format("binaryFile").load(imagePath)

val binaryToImage = new BinaryToImage()
  .setInputCol("content")
  .setOutputCol("image")

val data = binaryToImage.transform(df)

data.select("image").show()
```

```python
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read.format("binaryFile").load(imagePath)

binaryToImage = BinaryToImage() \
  .setInputCol("content") \
  .setOutputCol("image")

data = binaryToImage.transform(df)

data.select("image").show()
```

### ImageBinarizer

`ImageBinarizer` transforms image to binary color schema by threshold.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| threshold | int | 170 |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | binarized_image | image struct ([Image schema](#image-schema)) |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.ImageBinarizer
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val binirizer = new ImageBinarizer()
  .setInputCol("image")
  .setOutputCol("binary_image")
  .setThreshold(100)

val data = binirizer.transform(df)

data.storeImage("binary_image")
```

```python
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath) \
  .asImage("image")

binirizer = ImageBinarizer() \
  .setInputCol("image") \
  .setOutputCol("binary_image") \
  .setThreshold(100)

data = binirizer.transform(df)

data.show()

```
**Original image:**

![original](/assets/images/ocr/text_with_noise.png)

**Binarized image with 100 threshold:**

![binarized](/assets/images/ocr/binarized.png)


### ImageAdaptiveThresholding

Compute a threshold mask image based on local pixel neighborhood and apply it to image.

Also known as adaptive or dynamic thresholding. The threshold value is
the weighted mean for the local neighborhood of a pixel subtracted by a constant.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| blockSize | int | 170 | Odd size of pixel neighborhood which is used to calculate the threshold value (e.g. 3, 5, 7, ..., 21, ...). |
| method | string | | Method used to determine adaptive threshold for local neighbourhood in weighted mean image. |
| offset | int | | Constant subtracted from weighted mean of neighborhood to calculate the local threshold value. Default offset is 0. |
| mode | string | | The mode parameter determines how the array borders are handled, where cval is the value when mode is equal to 'constant' |
| cval | int | | Value to fill past edges of input if mode is 'constant'. |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | binarized_image | image struct ([Image schema](#image-schema)) |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
// Implemented only for Python































```

```python
from pyspark.ml import PipelineModel

from sparkocr.transformers import *
from sparkocr.utils import display_image

imagePath = "path to image"

# Read image file as binary file
df = spark.read 
    .format("binaryFile")
    .load(imagePath)

binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image")

adaptive_thresholding = ImageAdaptiveThresholding() \
    .setInputCol("scaled_image") \
    .setOutputCol("binarized_image") \
    .setBlockSize(21) \
    .setOffset(73)

pipeline = PipelineModel(stages=[
            binary_to_image,
            adaptive_thresholding
        ])

result = pipeline.transform(df)

for r in result.select("image", "corrected_image").collect():
    display_image(r.image)
    display_image(r.corrected_image)
```
**Original image:**

![original](/assets/images/ocr/text_with_noise.png)

**Binarized image:**

![binarized](/assets/images/ocr/adaptive_binarized.png)

### ImageErosion

`ImageErosion` erodes image.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| kernelSize | int | 2 |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | eroded_image | image struct ([Image schema](#image-schema)) |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.ImageErosion
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageErosion()
  .setInputCol("image")
  .setOutputCol("eroded_image")
  .setKernelSize(1)

val data = transformer.transform(df)
data.storeImage("eroded_image")
```

```python
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath) \
  .asImage("image")

transformer = ImageErosion() \
  .setInputCol("image") \
  .setOutputCol("eroded_image") \
  .setKernelSize(1)

data = transformer.transform(df)
data.show()

```

### ImageScaler

`ImageScaler` scales image by provided scale factor.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| scaleFactor | double | 1.0 | scale factor |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | scaled_image | scaled image struct ([Image schema](#image-schema)) |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.ImageScaler
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageScaler()
  .setInputCol("image")
  .setOutputCol("scaled_image")
  .setScaleFactor(0.5)

val data = transformer.transform(df)
data.storeImage("scaled_image")
```

```python
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath) \
  .asImage("image")

transformer = ImageScaler() \
  .setInputCol("image") \
  .setOutputCol("scaled_image") \
  .setScaleFactor(0.5)

data = transformer.transform(df)
data.show()

```

### ImageAdaptiveScaler

`ImageAdaptiveScaler` detects font size and scales image for have desired font size.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| desiredSize | int | 34 | desired size of font in pixels |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | scaled_image | scaled image struct ([Image schema](#image-schema)) |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.ImageAdaptiveScaler
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageAdaptiveScaler()
  .setInputCol("image")
  .setOutputCol("scaled_image")
  .setDesiredSize(34)

val data = transformer.transform(df)
data.storeImage("scaled_image")
```

```python
from sparkocr.transformers import *

imagePath = "path to image"

# Read image file as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath) \
  .asImage("image")

transformer = ImageAdaptiveScaler() \
  .setInputCol("image") \
  .setOutputCol("scaled_image") \
  .setDesiredSize(34)

data = transformer.transform(df)
data.show()

```

### ImageSkewCorrector

`ImageSkewCorrector` detects skew of the image and rotates it.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| rotationAngle | double | 0.0 | rotation angle |
| automaticSkewCorrection | boolean | true | enables/disables adaptive skew correction |
| halfAngle | double | 5.0 | half the angle(in degrees) that will be considered for correction |
| resolution | double | 1.0 | The step size(in degrees) that will be used for generating correction angle candidates |


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | corrected_image | corrected image struct ([Image schema](#image-schema)) |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.ImageSkewCorrector
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageSkewCorrector()
  .setInputCol("image")
  .setOutputCol("corrected_image")
  .setAutomaticSkewCorrection(true)

val data = transformer.transform(df)
data.storeImage("corrected_image")
```

```python
from sparkocr.transformers import *

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageSkewCorrector()
  .setInputCol("image")
  .setOutputCol("corrected_image")
  .setAutomaticSkewCorrection(true)

val data = transformer.transform(df)
data.show()

```

**Original image:**

![original](/assets/images/ocr/rotated.png)

**Corrected image:**

![corrected](/assets/images/ocr/corrected.png)

### ImageNoiseScorer

`ImageNoiseScorer` computes noise score for each region.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |
| inputRegionsCol | string | regions | regions |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| method | [NoiseMethod](#noisemethod) string | NoiseMethod.RATIO | method of computation noise score |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | noisescores | noise score for each region |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageNoiseScorer, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.NoiseMethod
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// Define transformer for detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("regions")

// Define transformer for compute noise level for each region
val noisescorer = new ImageNoiseScorer()
  .setInputCol("image")
  .setOutputCol("noiselevel")
  .setInputRegionsCol("regions")
  .setMethod(NoiseMethod.VARIANCE)

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  layoutAnalyzer,
  noisescorer
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = modelPipeline.transform(df)

data.select("path", "noiselevel").show()
```

```python
from pyspark.ml import PipelineModel

from sparkocr.transformers import *
from sparkocr.enums import NoiseMethod

imagePath = "path to image"

# Read image file as binary file
df = spark.read \
  .format("binaryFile") \
  .load(imagePath) \
  .asImage("image")

# Define transformer for detect regions
layoutAnalyzer = ImageLayoutAnalyzer() \
  .setInputCol("image") \
  .setOutputCol("regions")

# Define transformer for compute noise level for each region
noisescorer = ImageNoiseScorer() \
  .setInputCol("image") \
  .setOutputCol("noiselevel") \
  .setInputRegionsCol("regions") \
  .setMethod(NoiseMethod.VARIANCE)

# Define pipeline
pipeline = Pipeline()
pipeline.setStages(Array(
  layoutAnalyzer,
  noisescorer
))

data = pipeline.transform(df)

data.select("path", "noiselevel").show()



```

**Output:**

```
+------------------+-----------------------------------------------------------------------------+
|path              |noiselevel                                                                   |
+------------------+-----------------------------------------------------------------------------+
|file:./noisy.png  |[32.01805641767766, 32.312916551193354, 29.99257352247787, 30.62470388308217]|
+------------------+-----------------------------------------------------------------------------+

```

### ImageRemoveObjects

**python only**

`ImageRemoveObjects` for remove background objects.
It support removing:
- objects less then elements of font with _minSizeFont_ size
- objects less then _minSizeObject_
- holes less then _minSizeHole_
- objects more then _maxSizeObject_

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | None | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| minSizeFont | int | 10 | Min size font in pt. |
| minSizeObject | int | None | Min size of object which will keep on image [*]. |
| connectivityObject | int | 0 | The connectivity defining the neighborhood of a pixel. |
| minSizeHole | int | None | Min size of hole which will keep on image[ *]. |
| connectivityHole | int | 0 | The connectivity defining the neighborhood of a pixel. |
| maxSizeObject | int | None | Max size of object which will keep on image [*]. |
| connectivityMaxObject | int | 0 | The connectivity defining the neighborhood of a pixel. |

[*] : _None_ value disables removing objects.

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | None | scaled image struct ([Image schema](#image-schema)) |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
// Implemented only for Python
























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

remove_objects = ImageRemoveObjects() \
    .setInputCol("image") \
    .setOutputCol("corrected_image") \
    .setMinSizeObject(20)

pipeline = PipelineModel(stages=[
    binary_to_image,
    remove_objects
])

data = pipeline.transform(df)
```

### ImageMorphologyOpening

**python only**

`ImageMorphologyOpening` Return greyscale morphological opening of an image.
                     
 The morphological opening on an image is defined as an erosion followed by
 a dilation. Opening can remove small bright spots (i.e. "salt") and connect
 small dark cracks. This tends to "open" up (dark) gaps between (bright)
 features.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | None | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| kernelShape | [KernelShape](#kernelshape) | KernelShape.DISK | Kernel shape. |
| kernelSize | int | 1 | Kernel size in pixels. |

[*] : _None_ value disables removing objects.

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | None | scaled image struct ([Image schema](#image-schema)) |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
// Implemented only for Python



































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

adaptive_thresholding = ImageAdaptiveThresholding() \
    .setInputCol("image") \
    .setOutputCol("corrected_image") \
    .setBlockSize(75) \
    .setOffset(0)

opening = ImageMorphologyOpening() \
    .setInputCol("corrected_image") \
    .setOutputCol("opening_image") \
    .setkernelSize(1)

pipeline = PipelineModel(stages=[
    binary_to_image,
    adaptive_thresholding,
    opening
])

result = pipeline.transform(df)

for r in result.select("image", "corrected_image").collect():
    display_image(r.image)
    display_image(r.corrected_image)
```

**Original image:**

![original](/assets/images/ocr/text_with_noise.png)

**Opening image:**

![opening](/assets/images/ocr/opening.png)

## Splitting image to regions

### ImageLayoutAnalyzer

`ImageLayoutAnalyzer` analyzes the image and determines regions of text.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| pageSegMode | [PageSegmentationMode](#pagesegmentationmode) | AUTO | page segmentation mode |
| pageIteratorLevel | [PageIteratorLevel](#pageiteratorlevel) | BLOCK | page iteration level |
| ocrEngineMode | [EngineMode](#enginemode) | LSTM_ONLY | OCR engine mode |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | region | array of [Coordinaties](#coordinate-schema)|

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// Define transformer for detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("regions")

val data = layoutAnalyzer.transform(df)

data.show()





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

# Define transformer for detect regions
layout_analyzer = ImageLayoutAnalyzer() \
  .setInputCol("image") \
  .setOutputCol("regions")

pipeline = PipelineModel(stages=[
    binary_to_image,
    layout_analyzer
])

data = pipeline.transform(df)
data.show()
```

### ImageSplitRegions

`ImageSplitRegions` splits image to regions.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |
| inputRegionsCol | string | region | array of [Coordinaties](#coordinate-schema)|


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| explodeCols | Array[string] | |Columns which need to explode |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | region_image | image struct ([Image schema](#image-schema)) |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// Define transformer for detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("regions")

val splitter = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("regions")
  .setOutputCol("region_image")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  layoutAnalyzer,
  splitter
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)
data.show()
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

# Define transformer for detect regions
layout_analyzer = ImageLayoutAnalyzer() \
  .setInputCol("image") \
  .setOutputCol("regions")

splitter = ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("regions")
  .setOutputCol("region_image")

# Define pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    layout_analyzer,
    splitter
])

data = pipeline.transform(df)

data.show()
```

### ImageDrawRegions

`ImageDrawRegions` draw regions to image.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |
| inputRegionsCol | string | region | array of [Coordinaties](#coordinate-schema)|


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| lineWidth | Int | 4 | Line width for draw rectangles |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image_with_regions | image struct ([Image schema](#image-schema)) |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// Define transformer for detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("regions")

val draw = new ImageDrawRegions()
  .setInputCol("image")
  .setRegionCol("regions")
  .setOutputCol("image_with_regions")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  layoutAnalyzer,
  draw
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)
data.show()
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

# Define transformer for detect regions
layout_analyzer = ImageLayoutAnalyzer() \
  .setInputCol("image") \
  .setOutputCol("regions")

draw = ImageDrawRegions() \
  .setInputCol("image") \
  .setRegionCol("regions") \
  .setOutputCol("image_with_regions")

# Define pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    layout_analyzer,
    draw
])

data = pipeline.transform(df)
data.show()

```

## Characters recognition

Next section describes the estimators for OCR

### TesseractOCR

`TesseractOCR` runs Tesseract OCR for input image, return recognized text
to _outputCol_ and positions with font size to 'positions' column.


#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| pageSegMode | [PageSegmentationMode](#pagesegmentationmode) | AUTO | page segmentation mode |
| pageIteratorLevel | [PageIteratorLevel](#pageiteratorlevel) | BLOCK | page iteration level |
| ocrEngineMode | [EngineMode](#enginemode) | LSTM_ONLY| OCR engine mode |
| language | string | eng | language |
| confidenceThreshold | int | 0 | Confidence threshold. |
| ignoreResolution | bool | true | Ignore resolution from metadata of image. |
| tesseractParams | array of strings | [] |Array of Tesseract params in key=value format. |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | text | recognized text |

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers.TesseractOCR
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new TesseractOCR()
  .setInputCol("image")
  .setOutputCol("text")
  .setTesseractParams(Array("preserve_interword_spaces=1"))

val data = transformer.transform(df)
print(data.select("text").collect()[0].text)









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

ocr = TesseractOCR() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setTesseractParams(["preserve_interword_spaces=1", ])

# Define pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    ocr
])

data = pipeline.transform(df)
data.show()
```

**Image:**

![image](/assets/images/ocr/corrected.png)

**Output:**

```
FOREWORD

Electronic design engineers are the true idea men of the electronic
industries. They create ideas and use them in their designs, they stimu-
late ideas in other designers, and they borrow and adapt ideas from
others. One could almost say they feed on and grow on ideas.

```

## Other

Next section describes the extra transformers

### PositionFinder

`PositionFinder` find position of input text entities in original document.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCols | string | image | Input annotations columns |
| pageMatrixCol | string | | Column name for Page Matrix schema |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| matchingWindow | int | 10 | Textual range to match in context, applies in both direction |
| windowPageTolerance | boolean | true | whether or not to increase tolerance as page number grows |
| padding | int | 5| padding for area |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | | Name of output column for store coordinates. |

**Example:**


{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.transformers._
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.util.io.ReadAs

val pdfPath = "path to pdf"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToText = new PdfToText()
  .setInputCol("content")
  .setOutputCol("text")
  .setSplitPage(false)

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val entityExtractor = new TextMatcher()
  .setInputCols("sentence", "token")
  .setEntities("test-chunks.txt", ReadAs.TEXT)
  .setOutputCol("entity")

val positionFinder = new PositionFinder()
  .setInputCols("entity")
  .setOutputCol("coordinates")
  .setPageMatrixCol("positions")
  .setMatchingWindow(10)
  .setPadding(2)

// Create pipeline
val pipeline = new Pipeline()
  .setStages(Array(
    pdfToText,
    documentAssembler,
    sentenceDetector,
    tokenizer,
    entityExtractor,
    positionFinder
  ))

val results = pipeline.fit(df).transform(df)

results.show()
```

```python
from pyspark.ml import Pipeline

from sparkocr.transformers import *
from sparknlp.annotator import *
from sparknlp.base import *

pdfPath = "path to pdf"

# Read PDF file as binary file
df = spark.read.format("binaryFile").load(pdfPath)

pdf_to_text = PdfToText() \
    .setInputCol("content") \
    .setOutputCol("text") \
    .setPageNumCol("page") \
    .setSplitPage(False)

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

entity_extractor = TextMatcher() \
    .setInputCols("sentence", "token") \
    .setEntities("./sparkocr/resources/test-chunks.txt", ReadAs.TEXT) \
    .setOutputCol("entity")

position_finder = PositionFinder() \
    .setInputCols("entity") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(10) \
    .setPadding(2)

pipeline = Pipeline(stages=[
    pdf_to_text,
    document_assembler,
    sentence_detector,
    tokenizer,
    entity_extractor,
    position_finder
])

results = pipeline.fit(df).transform(df)
results.show()

```

# Structures and helpers

## OCR Schemas

### Image Schema

Images are loaded as a DataFrame with a single column called “image.” 

It is a struct-type column, that contains all information about image:

```
image: struct (nullable = true)
 |    |-- origin: string (nullable = true)
 |    |-- height: integer (nullable = false)
 |    |-- width: integer (nullable = false)
 |    |-- nChannels: integer (nullable = false)
 |    |-- mode: integer (nullable = false)
 |    |-- resolution: integer (nullable = true)
 |    |-- data: binary (nullable = true)
```

#### Fields

| Field name | Type | Description |
| --- | --- | --- |
| origin | string | source URI  |
| height | integer | image height in pixels |
| width | integer | image width in pixels |
| nChannels | integer | number of color channels |
| mode | [ImageType](#imagetype) | the data type and channel order the data is stored in |
| resolution | integer | Resolution of image in dpi |
| data | binary | image data in a binary format |


NOTE: Image `data` stored in a binary format. Image data is represented
      as a 3-dimensional array with the dimension shape (height, width, nChannels)
      and array values of type t specified by the mode field.
{:.info}


### Coordinate Schema

```
element: struct (containsNull = true)
 |    |    |-- index: integer (nullable = false)
 |    |    |-- page: integer (nullable = false)
 |    |    |-- x: float (nullable = false)
 |    |    |-- y: float (nullable = false)
 |    |    |-- width: float (nullable = false)
 |    |    |-- height: float (nullable = false)
```

| Field name | Type | Description |
| --- | --- | --- |
| index | integer | Chunk index |
| page | integer | Page number |
| x | float | The lower left x coordinate |
| y | float |  The lower left y coordinate |
| width | float |  The width of the rectangle |
| height | float |  The height of the rectangle |


## Enums

### PageSegmentationMode

  * ***OSD_ONLY***: Orientation and script detection only.
  * ***AUTO_OSD***: Automatic page segmentation with orientation and script detection.
  * ***AUTO_ONLY***: Automatic page segmentation, but no OSD, or OCR.
  * ***AUTO***: Fully automatic page segmentation, but no OSD.
  * ***SINGLE_COLUMN***: Assume a single column of text of variable sizes.
  * ***SINGLE_BLOCK_VERT_TEXT***: Assume a single uniform block of vertically aligned text.
  * ***SINGLE_BLOCK***: Assume a single uniform block of text.
  * ***SINGLE_LINE***: Treat the image as a single text line.
  * ***SINGLE_WORD***: Treat the image as a single word.
  * ***CIRCLE_WORD***: Treat the image as a single word in a circle.
  * ***SINGLE_CHAR***: Treat the image as a single character.
  * ***SPARSE_TEXT***: Find as much text as possible in no particular order.
  * ***SPARSE_TEXT_OSD***: Sparse text with orientation and script detection.

### EngineMode

  *  ***TESSERACT_ONLY***: Legacy engine only.
  *  ***OEM_LSTM_ONLY***: Neural nets LSTM engine only.
  *  ***TESSERACT_LSTM_COMBINED***: Legacy + LSTM engines.
  *  ***DEFAULT***: Default, based on what is available.
  
### PageIteratorLevel

  * ***BLOCK***: Block of text/image/separator line.
  * ***PARAGRAPH***: Paragraph within a block.
  * ***TEXTLINE***: Line within a paragraph.
  * ***WORD***: Word within a text line.
  * ***SYMBOL***: Symbol/character within a word.

### ImageType
 
 * ***TYPE_BYTE_GRAY***
 * ***TYPE_BYTE_BINARY***
 * ***TYPE_3BYTE_BGR***
 * ***TYPE_4BYTE_ABGR***
 
### NoiseMethod

 * ***VARIANCE***
 * ***RATIO***
 
### KernelShape

 * ***SQUARE***
 * ***DIAMOND***
 * ***DISK***
 * ***OCTAHEDRON***
 * ***OCTAGON***
 * ***STAR***
 
## OCR implicits

### asImage

`asImage` transforms binary content to [Image schema](#image-schema).

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| outputCol | string | image | output column name |
| contentCol | string | content | input column name with binary content |
| pathCol | string | path | input column name with path to original file |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

df.show()
```

```python
# Implemented only for Scala










```

### storeImage

`storeImage` stores the image(s) to tmp location and return Dataset with path(s) to stored image files.

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| inputColumn | string | | input column name with image struct |
| formatName | string | png | image format name |
| prefix | string | sparknlp_ocr_ | prefix for output file |


**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```scala
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// Read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

df.storeImage("image")
```

```python
# Implemented only for Scala










```

# Advanced Topics

## Error Handling

Pipeline execution would not be interrupted in case of the runtime exceptions 
while processing some records. 

In this case OCR transformers would fill _exception_ column that contains _transformer name_ and _exception_.

NOTE: Storing runtime errors to the _exception_ field allows to process batch of files. 
{:.info}


#### Output

Here is an output with exception when try to process js file using OCR pipeline:

{% include programmingLanguageSelectScalaPython.html %}

```scala
result.select("path", "text", "exception").show(2, false)
```

```python
result.select("path", "text", "exception").show(2, False)
```

```
+----------------------+-------------------------------------------+-----------------------------------------------------+
|path                  |text                                       |exception                                            |
+----------------------+-------------------------------------------+-----------------------------------------------------+
|file:jquery-1.12.3.js |                                           |BinaryToImage_c0311dc62161: Can't open file as image.|
|file:image.png        |I prefer the morning flight through Denver |null                                                 |
+----------------------+-------------------------------------------+-----------------------------------------------------+
```

## Performance

In case of big count of text PDF's in dataset
need have manual partitioning for avoid skew in partitions and effective utilize resources. 
For example the randomization could be used.
