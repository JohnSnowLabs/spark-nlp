---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.14.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_14_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

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

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}