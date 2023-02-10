---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.10.0
permalink: /docs/en/spark_ocr_versions/release_notes_1_10_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.10.0

Release date: 20-01-2021

#### Overview

Support Microsoft Docx documents.

#### New Features

* Added [DocToText](ocr_pipeline_components#doctotext) transformer for extract text
from DOCX documents.
* Added [DocToTextTable](ocr_pipeline_components#doctotexttable) transformer for extract
table data from DOCX documents.
* Added [DocToPdf](ocr_pipeline_components#doctopdf) transformer for convert DOCX
 documents to PDF format.

#### Bugfixes

* Fixed issue with loading model data on some cluster configurations


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}