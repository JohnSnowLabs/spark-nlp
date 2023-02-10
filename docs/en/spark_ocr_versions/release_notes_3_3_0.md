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

<div class="h3-box" markdown="1">

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

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}