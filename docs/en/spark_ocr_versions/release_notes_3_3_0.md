---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.3.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_3_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

## 3.3.0

Release date: 14-06-2021

#### Overview

Table detection and recognition for scanned documents.

For table detection we added ___ImageTableDetector___. 
It's based on __CascadeTabNet__ which used _Cascade mask Region-based CNN High-Resolution Network_ (Cascade mask R-CNN HRNet).
The model was pre-trained on the __COCO dataset__ and fine-tuned on __ICDAR 2019__ competitions dataset for table detection. It demonstrates state of the art results for ICDAR 2013 and TableBank. And top results for ICDAR 2019.

More details please read in [Table Detection & Extraction in Spark OCR](https://medium.com/spark-nlp/table-detection-extraction-in-spark-ocr-50765c6cedc9)

#### New Features

* [ImageTableDetector](ocr_table_recognition#imagetabledetector) is a DL model for detect tables on the image.
* [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) is a transformer for detect regions of cells in the table image.
* [ImageCellsToTextTable](ocr_table_recognition#imagecellstotexttable) is a transformer for extract text from the detected cells.

#### New notebooks

* [Image Table Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.3.0/jupyter/SparkOcrImageTableDetection.ipynb)
* [Image Cell Recognition example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.3.0/jupyter/SparkOcrImageTableCellRecognition.ipynb)
* [Image Table Recognition](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.3.0/jupyter/SparkOcrImageTableRecognition.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_0">Version 3.2.0</a>
    </li>
    <li>
        <strong>Version 3.3.0</strong>
    </li>
    <li>
        <a href="release_notes_3_4_0">Version 3.4.0</a>
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
  <li class="active"><a href="release_notes_3_3_0">3.3.0</a></li>
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