---
layout: model
title: Ocr pipeline in streaming
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
# Transform binary to image
pdf_to_image = PdfToImage() \
    .setInputCol("content") \
    .setOutputCol("image")

# Run OCR for each region
ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setConfidenceThreshold(60)

# OCR pipeline
pipeline = PipelineModel(stages=[
                            pdf_to_image,
                            ocr])

# fill path to folder with PDF's here
dataset_path = "data/pdfs/*.pdf"
# read one file for infer schema
pdfs_df = spark.read.format("binaryFile").load(dataset_path).limit(1)

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

# show results
spark.table("result").select("timestamp","pagenum", "path", "text").show(10)
```
```scala
# Transform binary to image
val pdf_to_image = new PdfToImage() 
    .setInputCol("content") 
    .setOutputCol("image")

# Run OCR for each region
val ocr = new ImageToText() 
    .setInputCol("image") 
    .setOutputCol("text") 
    .setConfidenceThreshold(60)

# OCR pipeline
val pipeline = new PipelineModel().setStages(Array(
    pdf_to_image, 
    ocr))

# fill path to folder with PDF's here
val dataset_path = "data/pdfs/*.pdf"
# read one file for infer schema
val pdfs_df = spark.read.format("binaryFile").load(dataset_path).limit(1)

# count of files in one microbatch
val maxFilesPerTrigger = 4 

# read files as stream
val pdf_stream_df = spark.readStream 
    .format("binaryFile") 
    .schema(pdfs_df.schema) 
    .option("maxFilesPerTrigger", maxFilesPerTrigger) 
    .load(dataset_path)

# process files using OCR pipeline
val result = pipeline.transform(pdf_stream_df).withColumn("timestamp", current_timestamp())

# store results to memory table
val query = result.writeStream 
    .format("memory") 
    .queryName("result") 
    .start()

# show results
spark.table("result").select(Array("timestamp","pagenum", "path", "text")).show(10)
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
