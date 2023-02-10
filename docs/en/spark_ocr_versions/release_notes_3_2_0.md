---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.2.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_2_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 3.2.0

Release date: 28-05-2021

#### Overview

Multi-modal visual document understanding, built on the LayoutLM architecture.
It achieves new state-of-the-art accuracy in several downstream tasks,
including form understanding and receipt understanding.


#### New Features

* [VisualDocumentNER](ocr_pipeline_components#visualdocumentner) is a DL model for NER problem using text and layout data.
  Currently available pre-trained model on the SROIE dataset.


#### Enhancements

* Added support `SPARK_OCR_LICENSE` env key for read license.
* Update dependencies and sync Spark versions with Spark NLP.


#### Bugfixes

* Fixed an issue that some ImageReaderSpi plugins are unavailable in the fat jar.

#### New notebooks

* [Visual Document NER](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.2.0/jupyter/SparkOCRVisualDocumentNer.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}