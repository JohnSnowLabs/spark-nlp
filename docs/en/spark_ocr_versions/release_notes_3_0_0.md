---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.0.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_0_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

## 3.0.0

Release date: 02-04-2021

#### Overview

We are very excited to release Spark OCR 3.0.0!

Spark OCR 3.0.0 extends the support for Apache Spark 3.0.x and 3.1.x major releases on Scala 2.12 with both Hadoop 2.7. and 3.2. We will support all 4 major Apache Spark and PySpark releases of 2.3.x, 2.4.x, 3.0.x, and 3.1.x.

Spark OCR started to support Tensorflow models. First model is [VisualDocumentClassifier](ocr_pipeline_components#visualdocumentclassifier).

#### New Features

* Support for Apache Spark and PySpark 3.0.x on Scala 2.12
* Support for Apache Spark and PySpark 3.1.x on Scala 2.12
* Support 9x new Databricks runtimes:
  * Databricks 7.3
  * Databricks 7.3 ML GPU
  * Databricks 7.4
  * Databricks 7.4 ML GPU
  * Databricks 7.5
  * Databricks 7.5 ML GPU
  * Databricks 7.6
  * Databricks 7.6 ML GPU
  * Databricks 8.0
  * Databricks 8.0 ML (there is no GPU in 8.0)
  * Databricks 8.1
* Support 2x new EMR 6.x: 
  * EMR 6.1.0 (Apache Spark 3.0.0 / Hadoop 3.2.1)
  * EMR 6.2.0 (Apache Spark 3.0.1 / Hadoop 3.2.1)
* [VisualDocumentClassifier](ocr_pipeline_components#visualdocumentclassifier) model for classification documents using text and layout data.
* Added support Vietnamese language.

#### New notebooks

* [Visual Document Classifier](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRVisualDocumentClassifier.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_1_11_0">Version 1.11.0</a>
    </li>
    <li>
        <strong>Version 3.0.0</strong>
    </li>
    <li>
        <a href="release_notes_3_1_0">Version 3.1.0</a>
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
  <li class="active"><a href="release_notes_3_0_0">3.0.0</a></li>
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