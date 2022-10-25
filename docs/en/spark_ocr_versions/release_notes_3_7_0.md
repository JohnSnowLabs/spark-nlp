---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.7.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_7_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

## 3.7.0

Release date: 30-08-2021

#### Overview

Improve table recognition and render OCR results to the PDF with original image


#### New Features

* Added [ImageToTextPdf](ocr_pipeline_components#imagetotextpdf) transformer for storing recognized text to the searchable
PDF with original image
* Added [PdfAssembler](ocr_pipeline_components#pdfassembler) for assembling multipage PDF document from single page PDF
documents


#### Enhancements

* Added support dbfs for store models. This allow to use models on Databricks.
* Improved [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added params for tuning [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added possibility to render detected lines to the original image in [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector)
* Added support to store recognized results to CSV in [ImageCellsToTextTable](ocr_table_recognition#imagecellstotexttable)
* Added [display_table](ocr_structures#displaytable) and [display_tables](ocr_structures#displaytables) functions
* Added [display_pdf_file](ocr_structures#displaypdffile) function for displaying pdf in embedded pdf viewer
* Updated license validator


#### New and updated notebooks

* [Process multiple page scanned PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrProcessMultiplepageScannedPDF.ipynb) (New)
* [Image Table Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableDetection.ipynb)
* [Image Cell Recognition example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableCellRecognition.ipynb)
* [Image Table Recognition](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableRecognition.ipynb)
* [Tables Recognition from PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableRecognitionPdf.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_6_0">Version 3.6.0</a>
    </li>
    <li>
        <strong>Version 3.7.0</strong>
    </li>
    <li>
        <a href="release_notes_3_8_0">Version 3.8.0</a>
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
  <li class="active"><a href="release_notes_3_7_0">3.7.0</a></li>
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