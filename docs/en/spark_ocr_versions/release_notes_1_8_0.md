---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.8.0
permalink: /docs/en/spark_ocr_versions/release_notes_1_8_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.8.0

Release date: 20-11-2020

#### Overview

Optimisation performance for processing multipage PDF documents.
Support up to 10k pages per document.

#### New Features

* Added [ImageAdaptiveBinarizer](ocr_pipeline_components#imageadaptivebinarizer) Scala transformer with support:
    - Gaussian local thresholding
    - Otsu thresholding
    - Sauvola local thresholding
* Added possibility to split pdf to small documents for optimize processing in [PdfToImage](ocr_pipeline_components#pdftoimage).


#### Enhancements

* Added applying binarization in [PdfToImage](ocr_pipeline_components#pdftoimage) for optimize memory usage.
* Added `pdfCoordinates` param to the [ImageToText](ocr_pipeline_components#imagetotext) transformer.
* Added 'total_pages' field to the [PdfToImage](ocr_pipeline_components#pdftoimage) transformer.
* Added different splitting strategies to the [PdfToImage](ocr_pipeline_components#pdftoimage) transformer.
* Simplified paging [PdfToImage](ocr_pipeline_components#pdftoimage) when run it with splitting to small PDF.
* Added params to the [PdfToText](ocr_pipeline_components#pdftotext) for disable extra functionality.
* Added `master_url` param to the python [start](ocr_install#using-start-function) function.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}