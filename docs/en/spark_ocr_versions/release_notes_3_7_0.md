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

<div class="h3-box" markdown="1">

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

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}