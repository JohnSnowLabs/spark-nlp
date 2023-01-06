---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.2.0
permalink: /docs/en/spark_ocr_versions/release_notes_1_2_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.2.0

Release date: 08-04-2020


#### Overview

Improved support Databricks and processing selectable pdfs.

#### Enhancements

* Adapted Spark OCR for run on Databricks.
* Added rewriting positions in [ImageToText](ocr_pipeline_components#imagetotext) when run together with PdfToText.
* Added 'positionsCol' param to [ImageToText](ocr_pipeline_components#imagetotext).
* Improved support Spark NLP. Changed [start](/ocr_install#using-start-function) function.

#### New Features

* Added [showImage](ocr_structures#showimages) implicit to Dataframe for display images in Scala Databricks notebooks.
* Added [display_images](ocr_structures#display_images) function for display images in Python Databricks notebooks.
* Added propagation selectable pdf file in [TextToPdf](ocr_pipeline_components#texttopdf). Added 'inputContent' param to 'TextToPdf'.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}