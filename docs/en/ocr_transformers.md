---
layout: article
title: Spark OCR Transformers
permalink: /docs/en/ocr_transformers
key: docs-ocr-transformers
modify_date: "2019-12-20"
---
Spark OCR provie set of Spark ML transformers/estimators for build OCR pipelines.

# OCR Pipelines

Using Spark OCR transformers possible to build pipelines for recognize text from:
 - image (png, tiff, jpeg ...)
 - selectable PDF
 - notselectable PDF

### OCR pipeline for image

### OCR pipeline for image PDF's

### OCR pipeline for text and image PDF's

# OCR Transformers

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
- setInputRegionsCol(string)
- setExplodeCols(string)

**Scala example:**

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
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

val splitter = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("region")
  .setOutputCol("region_image")

// define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  layoutAnalayzer,
  splitter
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)
data.show()
```

## OCR

Estimators for OCR

### ImageLayoutAnalyzer

Analyze image and determine regions of text.

**Settable parameters are:**

- setInputCol(string = "image")
- setOutputCol(string = "region")
- setPageSegMode(int = [PageSegmentationMode](#pagesegmentationmode).AUTO) - Page segmentation mode.
- setPageIteratorLevel(int = [PageIteratorLevel](#pageiteratorlevel).BLOCK) - Page iteration level.
- setOcrEngineMode(int = [EngineMode](#enginemode).LSTM_ONLY) - Ocr engine mode.

**Scala example:**

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
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

val data = layoutAnalayzer.transform(df)
data.show()
```

### TesseractOCR

Run Tesseract OCR for input image.

**Settable parameters are:**

- setInputCol(string = "image")
- setOutputCol(string = "text")
- setPageSegMode(int = [PageSegmentationMode](#pagesegmentationmode).AUTO) - Page segmentation mode.
- setPageIteratorLevel(int = [PageIteratorLevel](#pageiteratorlevel).BLOCK) - Page iteration level.
- setOcrEngineMode(int = [EngineMode](#enginemode).LSTM_ONLY) - Ocr engine mode.
- setLanguage(string = "eng") - Language.

**Scala example:**

```scala
import com.johnsnowlabs.ocr.transformers.TesseractOCR
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"

// read image file as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)
  .asImage("image")

val transformer = new TesseractOCR()
  .setInputCol("image")
  .setOutputCol("text")

val data = transformer.transform(df)
print(data.select("text").collect()[0].text)
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

## Ocr implicits

### asImage

Transform binary content to Image schema.

### storeImage

Store image to tmp location.

## ErrorHandling

Ocr ransformers fill exception column in case runtime exception. This allow
to process batch of files and do not interrupt processing when happen exception during
processing one record.

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

## OCR schemas

### Image

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