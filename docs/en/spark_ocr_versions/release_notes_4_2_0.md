---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 4.2.0
permalink: /docs/en/spark_ocr_versions/release_notes_4_2_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: spark-ocr
---

<div class="h3-box" markdown="1">

## 4.2.0

Release date: 31-10-2022


We are glad to announce that Spark OCR 4.2.0 has been released. This is mostly a compatibility release to ensure compatibility of Spark OCR against Spark NLP 4.2.1, and Spark NLP Healthcare 4.2.1.

#### Improvements
* Improved memory consumption and performance in the training of Visual NER models.

#### New Features
* PdfToForm new param: useFullyQualifiedName, added capability to return fully qualified key names.

#### New or Updated Notebooks
* [SparkOcrProcessMultiplepageScannedPDF.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrProcessMultiplepageScannedPDF.ipynb) has been added to show how to serve a multi-page document processing pipeline.
* [SparkOcrDigitalFormRecognition.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/4.2.0-release-candidate/jupyter/FormRecognition/SparkOcrDigitalFormRecognition.ipynb) has been updated to show utilization of useFullyQualifiedName parameter.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}
