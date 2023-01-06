---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.13.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_13_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 3.13.0

Release date: 25-05-2022

We are glad to announce that Spark OCR 3.13.0 has been released!.
This release focuses around VisualDocumentNer models, adding ability to fine-tune, fixing bugs, and to leverage the Annotation Lab to generate training data.

#### New Features

* VisualDocumentNerV21:
  * Now you can fine tune models VisualDocumentNerV21 models on your own dataset.
  
* AlabReaders: 
  * New class to allow training data from the Annotation Lab to be imported into Spark OCR. Currently, the reader supports Visual Ner only.


#### Bug Fixes

* Feature extraction on VisualDocumentNer has been improved.

#### New notebooks

* [SparkOcrFormRecognitionFineTuning.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.13.0-release-candidate/jupyter/FormRecognition/SparkOcrFormRecognitionFineTuning.ipynb), end to end example on Visual Document Ner Fine-Tuning.
* [Databricks notebooks](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/databricks) on Github Spark-OCR Workshop repository have been updated, and fixed.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}