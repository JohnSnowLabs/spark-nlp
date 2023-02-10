---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.9.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_9_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 3.9.0

Release date: 20-10-2021

#### Overview

Improve visualization and support Spark NLP. 

#### New Features

* Added [HocrTokenizer](ocr_pipeline_components#hocrtokenizer)
* Added [HocrDocumentAssembler](ocr_pipeline_components#hocrdocumentassembler)
* Added [ImageDrawAnnotations](ocr_pipeline_components#imagedrawannotations)
* Added support Arabic language in ImageToText and ImageToHocr

#### Enhancements

* Added postprocessing to the [ImageTableDetector](ocr_table_recognition#imagetabledetector)
* Added Spark NLP by default to spark session in start function
* Changed default value for ignoreResolution param in [ImageToText](ocr_pipeline_components#imagetotext)
* Updated license-validator. Added support floating license and set AWS keys from license.
* Added 'whiteList' param to the [VisualDocumentNER](ocr_pipeline_components#visualdocumentner)

#### New and updated notebooks

* [Spark OCR HOCR](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.9.0/jupyter/SparkOcrHocr.ipynb)
* [Visual Document NER](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.9.0/jupyter/SparkOCRVisualDocumentNer.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}