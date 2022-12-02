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


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_1_0_0">Version 1.0.0</a>
    </li>
    <li>
        <strong>Version 1.1.0</strong>
    </li>
    <li>
        <a href="release_notes_1_1_1">Version 1.1.1</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_3_14_0">3.14.0</a></li>
  <li><a href="release_notes_3_13_0">3.13.0</a></li>
  <li><a href="release_notes_3_12_0">3.12.0</a></li>
  <li><a href="release_notes_3_11_0">3.11.0</a></li>
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
  <li class="active"><a href="release_notes_1_1_0">1.1.0</a></li>
  <li><a href="release_notes_1_0_0">1.0.0</a></li>
</ul>