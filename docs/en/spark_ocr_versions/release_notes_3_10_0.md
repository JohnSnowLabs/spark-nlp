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

<div class="h3-box" markdown="1">

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

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}