---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.3.0
permalink: /docs/en/spark_ocr_versions/release_notes_1_3_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.3.0

Release date: 22-05-2020

#### Overview

New functionality for de-identification problem.

#### Enhancements

* Renamed TesseractOCR to ImageToText. 
* Simplified installation.
* Added check license from `SPARK_NLP_LICENSE` env varibale.

#### New Features

* Support storing for binaryFormat. Added support storing Image and PDF files.
* Support selectable pdf for [TextToPdf](ocr_pipeline_components#texttopdf) transformer.
* Added [UpdateTextPosition](ocr_pipeline_components#updatetextposition) transformer.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}