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

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_10_0">Version 3.10.0</a>
    </li>
    <li>
        <strong>Version 3.11.0</strong>
    </li>
    <li>
        <a href="release_notes_3_12_0">Version 3.12.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_3_14_0">3.14.0</a></li>
  <li><a href="release_notes_3_13_0">3.13.0</a></li>
  <li><a href="release_notes_3_12_0">3.12.0</a></li>
  <li class="active"><a href="release_notes_3_11_0">3.11.0</a></li>
  <li><a href="release_notes_3_10_0">3.10.0</a></li>
  <li><a href="release_notes_3_9_1">3.9.1</a></li>
  <li><a href="release_notes_3_9_0">3.9.0</a></li>
  <li><a href="release_notes_3_8_0">3.8.0</a></li>
  <li><a href="release_notes_3_7_0">3.7.0</a></li>
  <li><a href="release_notes_3_6_0">3.6.0</a></li>
  <li><a href="release_notes_3_5_0">3.5.0</a></li>
  <li><a href="release_notes_3_4_0">3.4.0</a></li>
  <li><a href="release_notes_3_3_0">3.3.0</a></li>
  <li><a href="release_notes_3_2_0">3.2.0</a></li>
  <li><a href="release_notes_3_1_0">3.1.0</a></li>
  <li><a href="release_notes_3_0_0">3.0.0</a></li>
  <li><a href="release_notes_1_11_0">1.11.0</a></li>
  <li><a href="release_notes_1_10_0">1.10.0</a></li>
  <li><a href="release_notes_1_9_0">1.9.0</a></li>
  <li><a href="release_notes_1_8_0">1.8.0</a></li>
  <li><a href="release_notes_1_7_0">1.7.0</a></li>
  <li><a href="release_notes_1_6_0">1.6.0</a></li>
  <li><a href="release_notes_1_5_0">1.5.0</a></li>
  <li><a href="release_notes_1_4_0">1.4.0</a></li>
  <li><a href="release_notes_1_3_0">1.3.0</a></li>
  <li><a href="release_notes_1_2_0">1.2.0</a></li>
  <li><a href="release_notes_1_1_2">1.1.2</a></li>
  <li><a href="release_notes_1_1_1">1.1.1</a></li>
  <li><a href="release_notes_1_1_0">1.1.0</a></li>
  <li><a href="release_notes_1_0_0">1.0.0</a></li>
</ul>