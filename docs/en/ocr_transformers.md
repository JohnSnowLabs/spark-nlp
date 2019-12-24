---
layout: article
title: Spark OCR Transformers
permalink: /docs/en/ocr_transformers
key: docs-ocr-transformers
modify_date: "2019-12-20"
---
Spark OCR provie set of Spark ML transformers/estimators for build OCR pipelines.

## OCR Pipelines

Using Spark OCR transformers possible to build pipelines for recognize text from:
 - image (png, tiff, jpeg ...)
 - selectable PDF
 - notselectable PDF

Spark OCR represent image as StructType with following schema:
```
image: struct (nullable = true)
 |-- origin: string (nullable = true)
 |-- height: integer (nullable = false)
 |-- width: integer (nullable = false)
 |-- nChannels: integer (nullable = false)
 |-- mode: integer (nullable = false)
 |-- data: binary (nullable = true)
```


## PDF processing

Transformers for deal with PDF files.

### PDFToText

Extract text from selectable PDF.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setPageNumCol(string)
- setOriginCol(string)
- setSplitPage(bool)

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.PdfToText

val pdfPath = "path to pdf with text layout"

// read pdf file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val transformer = new PdfToText()
  .setInputCol("content")
  .setOutputCol("text")
  .setPageNumCol("pagenum")
  .setSplitPage(true)

val data = transformer.transform(df)

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


### PDFToImage

Render PDF to image.

**Settable parameters are:**

- setInputCol()
- setOutputCol()
- setPageNumCol()
- setOriginCol()
- setSplitPage()
- setMinSizeBeforeFallback()
- setFallBackCol()

**Example:**

**Scala**

```scala
import com.johnsnowlabs.ocr.transformers.PDFToImage

val pdfPath = "path to pdf"

// read pdf file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToImage = new PDFToImage()
  .setInputCol("content")
  .setOutputCol("text")
  .setPageNumCol("pagenum")
  .setSplitPage(true)

val data =  pdfToImage.transform(df)

data.select("pagenum", "text").show()
```

**Python**

```python
pdfToImage = PDFToImage()
```

## Image processing

Transformers for image preprocessing.

### BinaryToImage

Transform image loaded as binary file to image struct.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setOriginCol(string)

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.BinaryToImage

val imagePath = "path to image"

// read image file as binary file
val df = spark.read.format("binaryFile").load(imagePath)

val binaryToImage = new BinaryToImage()
  .setInputCol("content")
  .setOutputCol("image")

val data = binaryToImage.transform(df)

data.select("image").show()
```

### ImageBinaryzer

Transform image to binary color schema by treshold.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setThreshold(int) - default: 170

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageBinaryzer
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val biniryzer = new ImageBinaryzer()
  .setInputCol("image")
  .setOutputCol("binary_image")
  .setThreshold(100)

val data = biniryzer.transform(df)
data.storeImage("binary_image")
```
**Original image:**

![original](/assets/images/ocr/text_with_noise.png)

**Binarized image with 100 treshold:**

![binarized](/assets/images/ocr/binarized.png)

### ImageErosion

Erdore image.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setKernelSize(int)

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageErosion
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
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

### ImageScaler

Scale image by provided scale factor.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setScaleFactor(double)

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageScaler
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
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

### ImageAdaptiveScaler

Detect font size and scale image for have desired font size.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setDesiredSize(int) - desired size of font in pixels

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageAdaptiveScaler
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
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

### ImageSkewCorrector

Detect skew of image and rotate image.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setRotationAngle(double)
- setAutomaticSkewCorrection(boolean)
- setHalfAngle(double)
- setResolution(double)


**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageSkewCorrector
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
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

**Original image:**

![original](/assets/images/ocr/rotated.png)

**Corrected image:**

![corrected](/assets/images/ocr/corrected.png)

### ImageNoiseScorer

Compute noise score for each region.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)
- setMethod(string)
- setInputRegionsCol(string)

**Scala example:**

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageNoiseScorer, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.NoiseMethod
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

// define transformer for detect regions
val layoutAnalayzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("regions")

// define transformer for compute noise level for each region
val noisescorer = new ImageNoiseScorer()
  .setInputCol("image")
  .setOutputCol("noiselevel")
  .setInputRegionsCol("regions")
  .setMethod(NoiseMethod.VARIANCE)

// define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  layoutAnalayzer,
  noisescorer
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = modelPipeline.transform(df)

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

### ImageSplitRegions

Split image to regions.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.ImageSplitRegions
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("region")
  .setOutputCol("region_image")

val data = transformer.transform(df)
data.show()
```

## OCR

Estimators for OCR

### ImageLayoutAnalyzer

Analyze image and determine regions of text.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)

### TesseractOCR

Run Tesseract OCR for input image.

**Settable parameters are:**

- setInputCol(string)
- setOutputCol(string)

## Ocr implicits

### asImage

Transform binary content to Image schema.

### storeImage

Store image to tmp location.

## ErrorHandling

Ocr ransformers fill exception column in case runtime exception. This allow
to process batch of files and do not interrupt processing when happen exception during
processing one record.