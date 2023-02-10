---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.1.0
permalink: /docs/en/spark_ocr_versions/release_notes_1_1_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.1.0

Release date: 03-03-2020

#### Overview

This release contains improvements for preprocessing image before run OCR and
added possibility to store results to PDF for keep original formatting.


#### New Features

* Added auto calculation maximum size of objects for removing in `ImageRemoveObjects`.
  This improvement avoids to remove `.` and affect symbols with dots (`i`, `!`, `?`).
  Added `minSizeFont` param to `ImageRemoveObjects` transformer for
  activate this functional.
* Added `ocrParams` parameter to `ImageToText` transformer for set any
  ocr params.
* Added extraction font size in `ImageToText`
* Added `TextToPdf` transformer for render text with positions to pdf file.


#### Enhancements

* Added setting resolution in `ImageToText`. And added `ignoreResolution` param with
  default `true` value to `ImageToText` transformer for back compatibility.
* Added parsing resolution from image metadata in `BinaryToImage` transformer.
* Added storing resolution in `PdfToImage` transformer.
* Added resolution field to Image schema.
* Updated 'start' function for set 'PYSPARK_PYTHON' env variable.
* Improve auto-scaling/skew correction:
   - improved access to images values
   - removing unnecessary copies of images
   - adding more test cases
   - improving auto-correlation in auto-scaling.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}