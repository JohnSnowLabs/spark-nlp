---
layout: article
title: Spark OCR 2.0 (Licensed)
permalink: /docs/en/ocr_transformers
key: docs-ocr-transformers
modify_date: "2020-02-03"
---
Spark OCR provides set of Spark ML transformers/estimators that help users create and use OCR pipelines. 
It built on top of Tesseract OCR.

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

# Quickstart Examples

## Images

In the following code example we will create OCR Pipeline for processing image(s). 
The image file(s) can contain complex layout like columns, tables, images inside.

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer, BinaryToImage}
import com.johnsnowlabs.ocr.transformers.TesseractOcr

val imagePath = "path to image files"

// Read image files as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)

// Transform binary content to image
val binaryToImage = new BinaryToImage()
  .setInputCol("content")
  .setOutputCol("image")

// Detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("region")

// Split to regions
val splitter = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("region")
  .setOutputCol("region_image")

// OCR
val ocr = new TesseractOcr()
  .setInputCol("region_image")
  .setOutputCol("text")

// Define Pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  binaryToImage,
  layoutAnalyzer,
  splitter,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)

data.show()
```

## Scanned PDF files

Next sample provides an example of OCR Pipeline for processing PDF files with image data.
In this case it needed to use [PdfToImage](#pdftoimage) transformer to convert PDF file to the set of images.

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.transformers.{PdfToImage,  TesseractOcr}

val imagePath = "path to pdf files"

// Read pdf files as binary file
val df = spark.read
  .format("binaryFile")
  .load(imagePath)

// Transform PDF file to the image
val pdfToImage = new PdfToImage()
  .setInputCol("content")
  .setOutputCol("image")

// Detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("region")

// Split to regions
val splitter = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("region")
  .setOutputCol("region_image")

// OCR
val ocr = new TesseractOcr()
  .setInputCol("region_image")
  .setOutputCol("text")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  pdfToImage,
  layoutAnalyzer,
  splitter,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)

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

```scala
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.ocr.transformers.{ImageSplitRegions, ImageLayoutAnalyzer}
import com.johnsnowlabs.ocr.transformers.{PdfToText, PdfToImage,  TesseractOcr}

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

// Detect regions
val layoutAnalyzer = new ImageLayoutAnalyzer()
  .setInputCol("image")
  .setOutputCol("region")

// Split to regions
val splitter = new ImageSplitRegions()
  .setInputCol("image")
  .setRegionCol("region")
  .setOutputCol("region_image")

// OCR
val ocr = new TesseractOcr()
  .setInputCol("region_image")
  .setOutputCol("text")

// Define pipeline
val pipeline = new Pipeline()
pipeline.setStages(Array(
  pdfToText,
  pdfToImage,
  layoutAnalyzer,
  splitter,
  ocr
))

val modelPipeline = pipeline.fit(spark.emptyDataFrame)

val data = pipeline.transform(df)

data.show()
```

## Images (streaming mode)

Next code segments provide an example of streaming OCR pipeline.
It processes images and stores results to memory table.

```scala
val imagePath = "path folder with images"

val batchDataFrame = spark.read.format("binaryFile").load(imagePath).limit(1)
    
val pipeline = new Pipeline()
pipeline.setStages(Array(
  binaryToImage,
  binarizer,
  layoutAnalyzer,
  splitter,
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

For getting results from memory table following code could be used:

```scala
spark.table("results").select("path", "text").show()
```

More details about Spark Structured Streaming could be found in 
[spark documentation](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).
{:.info}

# Pipeline components

## PDF processing

Next section describes the transformers for deal with PDF files: extracting text and image data from PDF
files.

### PDFToText

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

`PDFToImage` renders PDF to an image. To be used with scanned PDF documents.

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


#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image | extracted image struct ([Image schema](#image-schema)) |
| pageNumCol | string | pagenum | page number or 0 when `splitPage = false` |


**Example:**

```scala
import com.johnsnowlabs.ocr.transformers.PDFToImage

val pdfPath = "path to pdf"

// Read PDF file as binary file
val df = spark.read.format("binaryFile").load(pdfPath)

val pdfToImage = new PDFToImage()
  .setInputCol("content")
  .setOutputCol("text")
  .setPageNumCol("pagenum")
  .setSplitPage(true)

val data =  pdfToImage.transform(df)

data.select("pagenum", "text").show()
```

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
**Original image:**

![original](/assets/images/ocr/text_with_noise.png)

**Binarized image with 100 threshold:**

![binarized](/assets/images/ocr/binarized.png)

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

**Output:**

```
+------------------+-----------------------------------------------------------------------------+
|path              |noiselevel                                                                   |
+------------------+-----------------------------------------------------------------------------+
|file:./noisy.png  |[32.01805641767766, 32.312916551193354, 29.99257352247787, 30.62470388308217]|
+------------------+-----------------------------------------------------------------------------+

```

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
  .setRegionCol("region")
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

## Characters recognition

Next section describes the estimators for OCR

### TesseractOCR

`TesseractOCR` runs Tesseract OCR for input image.

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

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | text | recognized text |

**Example:**

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

### storeImage

`storeImage` stores the image(s) to tmp location and return Dataset with path(s) to stored image files.

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| inputColumn | string | | input column name with image struct |
| formatName | string | png | image format name |
| prefix | string | sparknlp_ocr_ | prefix for output file |


**Example:**

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

# Advanced Topics

## Error Handling

Pipeline execution would not be interrupted in case of the runtime exceptions 
while processing some records. 

In this case OCR transformers would fill _exception_ column that contains _transformer name_ and _exception_.

NOTE: Storing runtime errors to the _exception_ field allows to process batch of files. 
{:.info}


#### Output

Here is an output with exception when try to process js file using OCR pipeline:

```scala
result.select("path", "text", "exception").show(2, false)
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
