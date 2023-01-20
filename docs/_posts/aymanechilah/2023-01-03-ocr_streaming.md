---
layout: model
title: ocr_streaming
author: John Snow Labs
name: ocr_streaming
date: 2023-01-03
tags: [en, licensed, ocr, streaming]
task: Ocr Streaming
language: en
edition: Visual NLP 4.0.0
spark_version: 3.0
supported: true
annotator: OcrStreaming
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Streaming pipeline implementation for the OCR task, using tesseract models. Tesseract is an Optical Character Recognition (OCR) engine developed by Google. It is an open-source tool that can be used to recognize text in images and convert it into machine-readable text. The engine is based on a neural network architecture and uses machine learning algorithms to improve its accuracy over time.

Tesseract has been trained on a variety of datasets to improve its recognition capabilities. These datasets include images of text in various languages and scripts, as well as images with different font styles, sizes, and orientations. The training process involves feeding the engine with a large number of images and their corresponding text, allowing the engine to learn the patterns and characteristics of different text styles. One of the most important datasets used in training Tesseract is the UNLV dataset, which contains over 400,000 images of text in different languages, scripts, and font styles. This dataset is widely used in the OCR community and has been instrumental in improving the accuracy of Tesseract. Other datasets that have been used in training Tesseract include the ICDAR dataset, the IIIT-HWS dataset, and the RRC-GV-WS dataset.

In addition to these datasets, Tesseract also uses a technique called adaptive training, where the engine can continuously improve its recognition capabilities by learning from new images and text. This allows Tesseract to adapt to new text styles and languages, and improve its overall accuracy.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/6.1.SparkOcrStreamingPDF.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from pyspark.ml import PipelineModel
from sparkocr.transformers import *

imagePath = "path to image"
pdfs_df = spark.read.format("binaryFile").load(imagePath)

# Transform binary to image
pdf_to_image = PdfToImage()
pdf_to_image.setOutputCol("image")

# Run OCR for each region
ocr = ImageToText()
        ocr.setInputCol("image")
        ocr.setOutputCol("text")
        ocr.setConfidenceThreshold(60)

# OCR pipeline
pipeline = PipelineModel(stages=[
                            pdf_to_image,
                            ocr])

# count of files in one microbatch
maxFilesPerTrigger = 4 

# read files as stream
pdf_stream_df = spark.readStream \
    .format("binaryFile") \
    .schema(pdfs_df.schema) \
    .option("maxFilesPerTrigger", maxFilesPerTrigger) \
    .load(dataset_path)

# process files using OCR pipeline
result = pipeline.transform(pdf_stream_df).withColumn("timestamp", current_timestamp())

# store results to memory table
query = result.writeStream \
   .format('memory') \
   .queryName('result') \
   .start()
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"
var batchDataFrame = spark.read.format("binaryFile").load(imagePath)

val binaryToImage = new BinaryToImage()
      .setInputCol("content")
      .setOutputCol("image")
  
val binarizer = new ImageBinarizer()
      .setInputCol("image")
      .setOutputCol("binarized_image")
  
val layoutAnalayzer = new ImageLayoutAnalyzer()
      .setInputCol("binarized_image")
      .setOutputCol("region")
      .setPageIteratorLevel(TessPageIteratorLevel.RIL_BLOCK)
  
val splitter = new ImageSplitRegions()
      .setInputCol("binarized_image")
      .setOutputCol("region_image")
  
val ocr = new ImageToText()
      .setInputCol("region_image")
      .setOutputCol("text")
      .setPageSegMode(6)
      .setPageIteratorLevel(TessPageIteratorLevel.RIL_BLOCK)
      
val pipeline = new Pipeline().setStages(Array(
                                          binaryToImage,
                                          binarizer,
                                          layoutAnalayzer,
                                          splitter,
                                          ocr
                                        ))

val modelPipeline = pipeline.fit(batchDataFrame)

val dataFrame = spark.readStream
    .format("binaryFile")
    .schema(batchDataFrame.schema)
    .load(imagePath)
  
val query = modelPipeline.transform(dataFrame)
    .select("text", "exception")
    .writeStream
    .format("memory")
    .queryName("test")
    .start()
  
query.processAllAvailable()
```
</div>

## Results

```bash
+--------------------+-------+--------------------+--------------------+
|           timestamp|pagenum|                path|                text|
+--------------------+-------+--------------------+--------------------+
|2022-07-20 21:45:...|      0|file:/content/dat...|FOREWORD\n\nElect...|
|2022-07-20 21:45:...|      0|file:/content/dat...|C nca Document fo...|
|2022-07-20 21:56:...|      0|file:/content/dat...|6/13/22, 11:47 AM...|
+--------------------+-------+--------------------+--------------------+
```
