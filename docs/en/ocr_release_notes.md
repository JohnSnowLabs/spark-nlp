---
layout: docs
header: true
seotitle: Spark OCR | John Snow Labs
title: Spark OCR release notes
permalink: /docs/en/ocr_release_notes
key: docs-ocr-release-notes
modify_date: "2020-04-08"
show_nav: true
sidebar:
    nav: spark-ocr
---


## 3.14.0

Release date: 13-06-2022

#### Overview

We are glad to announce that Spark OCR 3.14.0 has been released!.
This release focuses around Visual Document Classification models, native Image Preprocessing on the JVM, and bug fixes.

#### New Features

* VisualDocumentClassifierv2:
  * New annotator for classifying documents based on multimodal(text + images) features.
  
* VisualDocumentClassifierv3: 
  * New annotator for classifying documents based on image features.
 
* ImageTransformer:
  * New transformer that provides different image transformations on the JVM. Supported transforms are Scaling, Adaptive Thresholding, Median Blur, Dilation, Erosion, and Object Removal.


#### New notebooks

+ [SparkOCRVisualDocumentClassifierv2.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRVisualDocumentClassifierv2.ipynb), example of Visual Document Classification using multimodal (text + visual) features.
+ [SparkOCRVisualDocumentClassifierv3.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRVisualDocumentClassifierv3.ipynb), example of Visual Document Classification using only visual features.
+ [SparkOCRCPUImageOperations.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRCPUImageOperations.ipynb), example of ImageTransformer.


<div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>

<ul class="pagination">
    <li>
        <a href="spark_ocr_versions/release_notes_3_13_0">Versions 3.13.0</a>
    </li>
    <li>
        <strong>Versions 3.14.0</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
  <li class="active"><a href="spark_ocr_versions/release_notes_3_14_0">3.14.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_13_0">3.13.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_12_0">3.12.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_11_0">3.11.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_10_0">3.10.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_9_1">3.9.1</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_9_0">3.9.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_8_0">3.8.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_7_0">3.7.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_6_0">3.6.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_5_0">3.5.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_4_0">3.4.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_3_0">3.3.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_2_0">3.2.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_1_0">3.1.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_0_0">3.0.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_11_0">1.11.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_10_0">1.10.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_9_0">1.9.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_8_0">1.8.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_7_0">1.7.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_6_0">1.6.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_5_0">1.5.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_4_0">1.4.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_3_0">1.3.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_2_0">1.2.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_2">1.1.2</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_1">1.1.1</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_0">1.1.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_0_0">1.0.0</a></li>
</ul>