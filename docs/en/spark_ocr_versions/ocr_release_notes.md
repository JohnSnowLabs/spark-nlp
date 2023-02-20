---
layout: docs
header: true
seotitle: Spark OCR | John Snow Labs
title: Spark OCR release notes
permalink: /docs/en/spark_ocr_versions/ocr_release_notes
key: docs-ocr-release-notes
modify_date: "2023-02-17"
show_nav: true
sidebar:
    nav: spark-ocr
---

<div class="h3-box" markdown="1">

## 4.3.1

Release date: 17-02-2023

We're glad to announce that Visual NLP 😎 4.3.1 has been released.

### Highlights
* ImageTextCleaner & ImageTableDetector have improved memory consumption.
* New Annotators supported in LightPipelines.
* Table extraction from Digital PDFs pipeline now entirely supported as a LightPipeline.

### ImageTextCleaner & ImageTableDetector improved memory consumption
* ImageTextCleaner & ImageTableDetector improved memory consumption: we reduced about 30% the memory consumption for this annotator making it more memory friendly and enabling running on memory constrained environments like Colab.

### New Annotators supported in LightPipelines
Now the following annotators are supported in LightPipelines,
* PdfToHocr,
* HocrTokenizer,
* ImageTableDetector,
* ImageScaler,
* HocrToTextTable,

### Table extraction from Digital PDFs pipeline now entirely supported as a LightPipeline.
* Our Table Extraction from digital PDFs pipeline now supports running as a LightPipeline, check the updated notebook: [SparkOCRPdfToTable.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRPdfToTable.ipynb)

This release is compatible with Spark NLP for Healthcare 4.3.0, and Spark NLP 4.3.0.


</div><div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>

{%- include docs-sparckocr-pagination.html -%}
