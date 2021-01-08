---
layout: docs
header: true
title: Structures and helpers
permalink: /docs/en/ocr_structures
key: docs-ocr-structures
modify_date: "2020-04-08"
---

<div class="h3-box" markdown="1">

## Schemas

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

</div>

#### Fields

{:.table-model-big}
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

## Coordinate Schema

```
element: struct (containsNull = true)
 |    |    |-- index: integer (nullable = false)
 |    |    |-- page: integer (nullable = false)
 |    |    |-- x: float (nullable = false)
 |    |    |-- y: float (nullable = false)
 |    |    |-- width: float (nullable = false)
 |    |    |-- height: float (nullable = false)
```

{:.table-model-big}
| Field name | Type | Description |
| --- | --- | --- |
| index | integer | Chunk index |
| page | integer | Page number |
| x | float | The lower left x coordinate |
| y | float |  The lower left y coordinate |
| width | float |  The width of the rectangle |
| height | float |  The height of the rectangle |

<div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

### EngineMode

  *  ***TESSERACT_ONLY***: Legacy engine only.
  *  ***OEM_LSTM_ONLY***: Neural nets LSTM engine only.
  *  ***TESSERACT_LSTM_COMBINED***: Legacy + LSTM engines.
  *  ***DEFAULT***: Default, based on what is available.
  
</div><div class="h3-box" markdown="1">

### PageIteratorLevel

  * ***BLOCK***: Block of text/image/separator line.
  * ***PARAGRAPH***: Paragraph within a block.
  * ***TEXTLINE***: Line within a paragraph.
  * ***WORD***: Word within a text line.
  * ***SYMBOL***: Symbol/character within a word.

</div><div class="h3-box" markdown="1">

### ImageType
 
 * ***TYPE_BYTE_GRAY***
 * ***TYPE_BYTE_BINARY***
 * ***TYPE_3BYTE_BGR***
 * ***TYPE_4BYTE_ABGR***

</div><div class="h3-box" markdown="1">

### NoiseMethod

 * ***VARIANCE***
 * ***RATIO***

</div><div class="h3-box" markdown="1">

### KernelShape

 * ***SQUARE***
 * ***DIAMOND***
 * ***DISK***
 * ***OCTAHEDRON***
 * ***OCTAGON***
 * ***STAR***

</div><div class="h3-box" markdown="1">

### MorphologyOperationType
    
 * ***OPENING***
 * ***CLOSING***
 * ***EROSION***
 * ***DILATION***

</div><div class="h3-box" markdown="1">

### CropSquareType

 * ***TOP_LEFT***
 * ***TOP_CENTER***
 * ***TOP_RIGHT***
 * ***CENTER_LEFT***
 * ***CENTER***
 * ***CENTER_RIGHT***
 * ***BOTTOM_LEFT***
 * ***BOTTOM_CENTER***
 * ***BOTTOM_RIGHT***

</div><div class="h3-box" markdown="1">

### SplittingStrategy

 * ***FIXED_NUMBER_OF_PARTITIONS***
 * ***FIXED_SIZE_OF_PARTITION***
 
</div><div class="h3-box" markdown="1">

### TresholdingMethod

 * ***GAUSSIAN***
 * ***OTSU***
 * ***SAUVOLA***
 
</div><div class="h3-box" markdown="1">

## OCR implicits

### asImage

`asImage` transforms binary content to [Image schema](#image-schema).

</div><div class="h3-box" markdown="1">

#### Parameters

{:.table-model-big}
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

</div><div class="h3-box" markdown="1">

### storeImage

`storeImage` stores the image(s) to tmp location and return Dataset with path(s) to stored image files.

</div><div class="h3-box" markdown="1">

#### Parameters

{:.table-model-big}
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

</div><div class="h3-box" markdown="1">

### showImages

Show images on Databrics notebook.

</div><div class="h3-box" markdown="1">

#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| field | string | image | input column name with image struct |
| limit | integer | 5 | count of rows for display  |
| width | string | "800" | width of image |
| show_meta | boolean | true | enable/disable displaying methadata of image |

</div><div class="h3-box" markdown="1">

## Databricks Python helpers

### display_images

Show images.

</div>

#### Parameters

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| field | string | image | input column name with image struct |
| limit | integer | 5 | count of rows for display  |
| width | string | "800" | width of image |
| show_meta | boolean | true | enable/disable displaying methadata of image |

**Example:**

```python
from sparkocr.databricks import display_images
from sparkocr.transformers import BinaryToImage

images_path = "/tmp/ocr/images/*.tif"
images_example_df = spark.read.format("binaryFile").load(images_path).cache()

display_images(BinaryToImage().transform(images_example_df), limit=3)
```

![image](/assets/images/ocr/showImage.png)