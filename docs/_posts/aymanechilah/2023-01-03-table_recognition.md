---
layout: model
title: table_recognition
author: John Snow Labs
name: general_model_table_detection_v2
date: 2023-01-03
tags: [en, licensed, ocr, table_recognition]
task: Table Recognition
language: en
edition: Visual NLP 4.1.0
spark_version: 3.3.0
supported: true
annotator: TableRecognition
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model shows the capabilities for table recognition and free-text extraction using OCR techniques.
For table recognition is proposed a CascadTabNet model. 
CascadeTabNet is a machine learning model for table detection in document images. It is based on a cascaded architecture, which is a two-stage process where the model first detects candidate regions that may contain tables, and then classifies these regions as tables or non-tables. The model is trained using a dataset of document images, where the tables have been manually annotated.

The benchmark results show that the model is able to detect tables in document images with high accuracy.
On the ICDAR2013 table competition dataset, CascadeTabNet achieved an F1-score of 0.85, which is considered a good score in this dataset. On the COCO-Text dataset, the model achieved a precision of 0.82 and a recall of 0.79, which are also considered good scores. In addition, the model has been evaluated on the UNLV dataset, where it achieved a precision of 0.87 and a recall of 0.83.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/2.2.Spark_OCR_training_Table_recognition.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/general_model_table_detection_v2_en_3.3.0_3.0_1623301511401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from pyspark.ml import PipelineModel
from sparkocr.transformers import *

imagePath = "path to image"
bin_df = spark.read.format("binaryFile").load(imagePath)

# TABLE DATA EXTRACTION
binary_to_image = BinaryToImage()

# Detect tables on the page using pretrained model
table_detector = ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
table_detector.setInputCol("image")
table_detector.setOutputCol("region")

# Draw detected region's with table to the page
draw_regions = ImageDrawRegions()
draw_regions.setInputCol("image")
draw_regions.setInputRegionsCol("region")
draw_regions.setOutputCol("image_with_regions")
draw_regions.setRectColor(Color.red)

# Extract table regions to separate images
splitter = ImageSplitRegions()
splitter.setInputCol("image")
splitter.setInputRegionsCol("region")
splitter.setOutputCol("table_image")
splitter.setDropCols("image")

# Detect cells on the table image
cell_detector = ImageTableCellDetector()
cell_detector.setInputCol("table_image")
cell_detector.setOutputCol("cells")
cell_detector.setAlgoType("morphops")
cell_detector.setDrawDetectedLines(True)

# Extract text from the detected cells 
table_recognition = ImageCellsToTextTable()
table_recognition.setInputCol("table_image")
table_recognition.setCellsCol('cells')
table_recognition.setMargin(3)
table_recognition.setStrip(True)
table_recognition.setOutputCol('table')

# Erase detected table regions
fill_regions = ImageDrawRegions()
fill_regions.setInputCol("image")
fill_regions.setInputRegionsCol("region")
fill_regions.setOutputCol("image_1")
fill_regions.setRectColor(Color.white)
fill_regions.setFilledRect(True)

# OCR
ocr = ImageToText()
ocr.setInputCol("image_1")
ocr.setOutputCol("text")
ocr.setOcrParams(["preserve_interword_spaces=1", ])
ocr.setKeepLayout(True)
ocr.setOutputSpaceCharacterWidth(8)

pipeline_table = PipelineModel(stages=[
    binary_to_image,
    table_detector,
    draw_regions,
    fill_regions,
    splitter,
    cell_detector,
    table_recognition,
    ocr
])

tables_results = pipeline_table.transform(df).cache()
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val pdfPath = "path to pdf"
var df1 = spark.read.format("binaryFile").load(pdfPath)

val pdf2hocr = new PdfToHocr()
  .setOutputCol("hocr")

var df2 = pdf2hocr.transform(df1).cache()
.filter(
  col("pagenum") === lit(1) ||
  col("pagenum") === lit(4)
)

val pdf2image = new PdfToImage()
  .setInputCol("content")
  .setOutputCol("image")
  .setImageType(ImageType.TYPE_3BYTE_BGR)
  .setPageNumCol("tmp_pagenum")
  
val tableDetector = ImageTableDetector
  .pretrained("general_model_table_detection_v2", "en", "public/ocr/models")
  .setInputCol("image")
  .setOutputCol("table_regions")
  .setScoreThreshold(0.9)
  .setApplyCorrection(true)
  .setScaleWidthToCol("width_dimension")
  .setScaleHeightToCol("height_dimension")
  
val hocr2table = new HocrToTextTable()
  .setInputCol("hocr")
  .setRegionCol("table_regions")
  .setOutputCol("tables")
  .setOutputFormat(TableOutputFormat.TABLE)
  
val pipeline = new Pipeline()
pipeline.setStages(Array(
  pdf2image,
  tableDetector,
  hocr2table
))

var data = pipeline.fit(df2).transform(df2).cache().drop("tmp_pagenum")

val rows = data.select("tables").collect()
```
</div>

## Result

![Screenshot](../../_examples_ocr/image9_out.png)

```bash
+--------------------+--------------------+--------------------+--------------------+-------+--------------------+--------------------+--------------------+------+--------------------+--------------------+--------------------+-----------------+---------+--------------------+--------------------+
|         table_image|              region|  image_with_regions|             image_1|pagenum|    modificationTime|                path|               image|length|               cells|        output_image|               table|       confidence|exception|                text|           positions|
+--------------------+--------------------+--------------------+--------------------+-------+--------------------+--------------------+--------------------+------+--------------------+--------------------+--------------------+-----------------+---------+--------------------+--------------------+
|{file:/content/cT...|{0, 0, 214.0, 437...|{file:/content/cT...|{file:/content/cT...|      0|2022-09-28 18:54:...|file:/content/cTD...|{file:/content/cT...|385071|[[[[0, 0, 408, 32...|{file:/content/cT...|{{0, 0, 0.0, 0.0,...|95.43954744611467|     null|                 ...|[{[{ , 0, 170.0, ...|
+--------------------+--------------------+--------------------+--------------------+-------+--------------------+--------------------+--------------------+------+--------------------+--------------------+--------------------+-----------------+---------+--------------------+--------------------+
```
