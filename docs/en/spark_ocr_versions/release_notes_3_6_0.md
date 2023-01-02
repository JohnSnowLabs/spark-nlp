---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.6.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_6_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 3.6.0

Release date: 05-08-2021

#### Overview

Handwritten detection and visualization improvement.


#### New Features

* Added [ImageHandwrittenDetector](ocr_object_detection#imagehandwrittendetector) for detecting 'signature', 'date', 'name',
 'title', 'address' and others handwritten text.
* Added rendering labels and scores in [ImageDrawRegions](ocr_pipeline_components#imagedrawregions).
* Added possibility to scale image to fixed size in [ImageScaler](ocr_pipeline_components#imagescaler)
 with keeping original ratio.


#### Enhancements

* Support new version of pip for installing python package
* Added support string labels for detectors
* Added an auto inferencing of the input shape for detector models
* New license validator


#### Bugfixes

* Fixed display BGR images in display functions


#### New and updated notebooks

* [Image Signature Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageSignatureDetection.ipynb)
* [Image Handwritten Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageHandwrittenDetection.ipynb)
* [Image Scaler example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageScaler.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}