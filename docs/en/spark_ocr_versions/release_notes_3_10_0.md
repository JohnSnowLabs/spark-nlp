---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.10.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_10_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

## 3.10.0

Release date: 10-01-2022


#### Overview

Form recognition using LayoutLMv2 and text detection.


#### New Features

* Added [VisualDocumentNERv2](ocr_visual_document_understanding#visualdocumentnerv2) transformer
* Added DL based [ImageTextDetector](ocr_object_detection#imagetextdetector) transformer
* Support rotated regions in [ImageSplitRegions](ocr_pipeline_components#imagesplitregions)
* Support rotated regions in [ImageDrawRegions](ocr_pipeline_components#imagedrawregions)


#### New Models

* LayoutLMv2 fine-tuned on FUNSD dataset
* Text detection model based on CRAFT architecture


#### New notebooks

* [Text Detection](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3100-release-candidate/jupyter/TextDetection/SparkOcrImageTextDetection.ipynb)
* [Visual Document NER v2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3100-release-candidate/jupyter/SparkOCRVisualDocumentNERv2.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_9_1">Version 3.9.1</a>
    </li>
    <li>
        <strong>Version 3.10.0</strong>
    </li>
    <li>
        <a href="release_notes_3_11_0">Version 3.11.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_3_14_0">3.14.0</a></li>
  <li><a href="release_notes_3_13_0">3.13.0</a></li>
  <li><a href="release_notes_3_12_0">3.12.0</a></li>
  <li><a href="release_notes_3_11_0">3.11.0</a></li>
  <li class="active"><a href="release_notes_3_10_0">3.10.0</a></li>
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