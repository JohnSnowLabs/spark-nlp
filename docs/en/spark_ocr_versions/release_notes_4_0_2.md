---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 4.0.2
permalink: /docs/en/spark_ocr_versions/release_notes_4_0_2
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 4.0.2

Release date: 12-09-2022

#### Overview

We are glad to announce that Spark OCR 4.0.2 has been released!
This release comes with new features, fixes and more!.


#### New Features

* VisualDocumentClassifierV2 is now trainable! Continuing with the effort to make all the most useful models easily trainable, we added training capabilities to this annotator.
* Added support for Simplified Chinese.
* Added new 'PdfToForm' annotator, capable of extracting forms from digital PDFs. This is different from previously introduced VisualDocumentNER annotator in that this new annotator works only on digital documents, as opposite to the scanned forms handled by VisualDocumentNER. PdfToForm is complementary to VisualDocumentNER.
 

#### Improvements

* Support for multi-frame dicom has been added.
* Added the missing load()​ method in ImageToTextV2.

 

#### New Notebooks

* We added two new notebooks for VisualDocumentClassifierV2, a [preprocessing notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/VisualDocumentClassifierTraining/Spark-ocr%20visual%20doc%20classifier%20v2%20preprocessing%20on%20databricks.ipynb), useful when you're dealing with large datasets, and a [fine-tuning notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/VisualDocumentClassifierTraining/SparkOCRVisualDocumentClassifierv2Training.ipynb).
* We added a [new sample notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/FormRecognition/SparkOcrDigitalFormRecognition.ipynb) showing how to extract forms from digital PDF documents.
* We added a [new sample notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/TextRecognition/SparkOcrImageToText-Chinese.ipynb) explaining how to use Simplified Chinese OCR.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}