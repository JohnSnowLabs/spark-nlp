---
layout: article
title: Pipeline components
permalink: /docs/en/ocr_pipeline_components
key: docs-ocr-pipeline-components
modify_date: "2020-04-08"
use_language_switcher: "Python-Scala-Java"
---

# PDF processing

Next section describes the transformers that deal with PDF files with the purpose of extracting text and image data from PDF files.

## PdfToText

`PDFToText` extracts text from selectable PDF (with text layout).

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | text | binary representation of the PDF document |
| originCol | string | path | path to the original file |

##### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| splitPage | bool | true | whether it needed to split document to pages |


##### Output Columns

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

## PdfToImage

`PdfToImage` renders PDF to an image. To be used with scanned PDF documents.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the PDF document |
| originCol | string | path | path to the original file |
| fallBackCol | string | text | extracted text from previous method for detect if need to run transformer as fallBack |


##### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| splitPage | bool | true | whether it needed to split document to pages |
| minSizeBeforeFallback | int | 10 | minimal count of characters to extract to decide, that the document is the PDF with text layout |
| imageType | [ImageType](ocr_structures#imagetype) | `ImageType.TYPE_BYTE_GRAY` | type of the image |
| resolution | int | 300 | Output image resolution in dpi |
| keepInput | boolean | false | Keep input column in dataframe. By default it is dropping. |
| partitionNum | int | 0 | Number of partitions (0 value - without repartition) |

##### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image | extracted image struct ([Image schema](ocr_structures#image-schema)) |
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

## ImageToPdf

`ImageToPdf` transform image to Pdf document.
If dataframe contains few records for same origin path, it groups image by origin
column and create multipage PDF document.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema))  |
| originCol | string | path | path to the original file |


##### Output Columns

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

## TextToPdf

`TextToPdf` renders ocr results to PDF document as text layout. Each symbol will render to same position
with same font size as in original image or PDF.
If dataframe contains few records for same origin path, it groups image by origin
column and create multipage PDF document.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | positions | column with positions struct  |
| inputImage | string | image | image struct ([Image schema](ocr_structures#image-schema))  |
| inputText | string | text | column name with recognized text |
| originCol | string | path | path to the original file |
| inputContent | string | content | column name with binary representation of original PDF file |


##### Output Columns

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

## PdfDrawRegions

`PdfDrawRegions` transformer for drawing regions to Pdf document.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the PDF document |
| originCol | string | path | path to the original file |
| inputRegionsCol | string | region | input column which contain regions |


##### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| lineWidth | integer | 1 | line width for draw regions |


##### Output Columns

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

# Image pre-processing

Next section describes the transformers for image pre-processing: scaling, binarization, skew correction, etc.

## BinaryToImage

`BinaryToImage` transforms image (loaded as binary file) to image struct.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | content | binary representation of the image |
| originCol | string | path | path to the original file |


##### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image | extracted image struct ([Image schema](ocr_structures#image-schema)) |

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

## ImageBinarizer

`ImageBinarizer` transforms image to binary color schema by threshold.

##### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

##### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| threshold | int | 170 |

##### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | binarized_image | image struct ([Image schema](ocr_structures#image-schema)) |

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


## ImageAdaptiveThresholding

Compute a threshold mask image based on local pixel neighborhood and apply it to image.

Also known as adaptive or dynamic thresholding. The threshold value is
the weighted mean for the local neighborhood of a pixel subtracted by a constant.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

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
| outputCol | string | binarized_image | image struct ([Image schema](ocr_structures#image-schema)) |

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

## ImageErosion

`ImageErosion` erodes image.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| kernelSize | int | 2 |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | eroded_image | image struct ([Image schema](ocr_structures#image-schema)) |


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

## ImageScaler

`ImageScaler` scales image by provided scale factor.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| scaleFactor | double | 1.0 | scale factor |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | scaled_image | scaled image struct ([Image schema](ocr_structures#image-schema)) |


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

## ImageAdaptiveScaler

`ImageAdaptiveScaler` detects font size and scales image for have desired font size.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| desiredSize | int | 34 | desired size of font in pixels |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | scaled_image | scaled image struct ([Image schema](ocr_structures#image-schema)) |

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

## ImageSkewCorrector

`ImageSkewCorrector` detects skew of the image and rotates it.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

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
| outputCol | string | corrected_image | corrected image struct ([Image schema](ocr_structures#image-schema)) |


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

## ImageNoiseScorer

`ImageNoiseScorer` computes noise score for each region.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |
| inputRegionsCol | string | regions | regions |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| method | [NoiseMethod](ocr_structures#noisemethod) string | NoiseMethod.RATIO | method of computation noise score |

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

## ImageRemoveObjects

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
| inputCol | string | None | image struct ([Image schema](ocr_structures#image-schema)) |

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
| outputCol | string | None | scaled image struct ([Image schema](ocr_structures#image-schema)) |


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

## ImageMorphologyOpening

**python only**

`ImageMorphologyOpening` Return greyscale morphological opening of an image.
                     
 The morphological opening on an image is defined as an erosion followed by
 a dilation. Opening can remove small bright spots (i.e. "salt") and connect
 small dark cracks. This tends to "open" up (dark) gaps between (bright)
 features.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | None | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| kernelShape | [KernelShape](ocr_structures#kernelshape) | KernelShape.DISK | Kernel shape. |
| kernelSize | int | 1 | Kernel size in pixels. |

[*] : _None_ value disables removing objects.

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | None | scaled image struct ([Image schema](ocr_structures#image-schema)) |


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

# Splitting image to regions

## ImageLayoutAnalyzer

`ImageLayoutAnalyzer` analyzes the image and determines regions of text.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| pageSegMode | [PageSegmentationMode](ocr_structures#pagesegmentationmode) | AUTO | page segmentation mode |
| pageIteratorLevel | [PageIteratorLevel](ocr_structures#pageiteratorlevel) | BLOCK | page iteration level |
| ocrEngineMode | [EngineMode](ocr_structures#enginemode) | LSTM_ONLY | OCR engine mode |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | region | array of [Coordinaties]ocr_structures#coordinate-schema)|

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

## ImageSplitRegions

`ImageSplitRegions` splits image to regions.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |
| inputRegionsCol | string | region | array of [Coordinaties]ocr_structures#coordinate-schema)|


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| explodeCols | Array[string] | |Columns which need to explode |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | region_image | image struct ([Image schema](ocr_structures#image-schema)) |

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

## ImageDrawRegions

`ImageDrawRegions` draw regions to image.

#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |
| inputRegionsCol | string | region | array of [Coordinaties]ocr_structures#coordinate-schema)|


#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| lineWidth | Int | 4 | Line width for draw rectangles |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | image_with_regions | image struct ([Image schema](ocr_structures#image-schema)) |

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

# Characters recognition

Next section describes the estimators for OCR

## TesseractOCR

`TesseractOCR` runs Tesseract OCR for input image, return recognized text
to _outputCol_ and positions with font size to 'positionsCol' column.


#### Input Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| inputCol | string | image | image struct ([Image schema](ocr_structures#image-schema)) |

#### Parameters

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| pageSegMode | [PageSegmentationMode](ocr_structures#pagesegmentationmode) | AUTO | page segmentation mode |
| pageIteratorLevel | [PageIteratorLevel](ocr_structures#pageiteratorlevel) | BLOCK | page iteration level |
| ocrEngineMode | [EngineMode](ocr_structures#enginemode) | LSTM_ONLY| OCR engine mode |
| language | string | eng | language |
| confidenceThreshold | int | 0 | Confidence threshold. |
| ignoreResolution | bool | true | Ignore resolution from metadata of image. |
| tesseractParams | array of strings | [] |Array of Tesseract params in key=value format. |

#### Output Columns

| Param name | Type | Default | Column Data Description |
| --- | --- | --- | --- |
| outputCol | string | text | Recognized text |
| positionsCol| string| positions | Positions of each block of text (related to `pageIteratorLevel`) | 

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

# Other

Next section describes the extra transformers

## PositionFinder

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
