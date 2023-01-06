---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.11.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_11_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 3.11.0

Release date: 28-02-2022


#### Overview

We are glad to announce that Spark OCR 3.11.0 has been released!.
This release comes with new models, new features, bug fixes, and notebook examples.

#### New Features

* Added [ImageTextDetectorV2](ocr_object_detection#imagetextdetectorv2) Python Spark-OCR Transformer for detecting printed and handwritten text
 using CRAFT architecture with Refiner Net.
* Added [ImageTextRecognizerV2](ocr_pipeline_components#imagetotextv2) Python Spark-OCR Transformer for recognizing
 printed and handwritten text based on Deep Learning Transformer Architecture.
* Added [FormRelationExtractor](ocr_visual_document_understanding#formrelationextractor) for detecting relations between key and value entities in forms.
* Added the capability of fine tuning VisualDocumentNerV2 models for key-value pairs extraction.

#### New Models

* ImageTextDetectorV2: this extends the ImageTextDetectorV1 character level text detection model with a refiner net architecture.
* ImageTextRecognizerV2: Text recognition for printed text based on the Deep Learning Transformer Architecture.

#### New notebooks

* [SparkOcrImageToTextV2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextRecognition/SparkOcrImageToTextV2.ipynb)
* [ImageTextDetectorV2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextDetection/SparkOcrImageTextDetectionV2.ipynb)
* [Visual Document NER v2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextRecognition/SparkOcrImageToTextV2.ipynb)
* [SparkOcrFormRecognition](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/FormRecognition/SparkOcrFormRecognition.ipynb)
* [SparkOCRVisualDocumentNERv2FineTune](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOCRVisualDocumentNERv2FineTune.ipynb)
* Creating Rest a API with Synapse to extract text from images, [SparkOcrRestApi](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOcrRestApi.ipynb)
* Creating Rest a API with Synapse to extract text from PDFs, [SparkOcrRestApiPdf](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOcrRestApiPdf.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}