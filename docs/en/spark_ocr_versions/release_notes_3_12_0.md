---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.12.0
permalink: /docs/en/spark_ocr_versions/release_notes_3_12_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

## 3.12.0

Release date: 14-04-2022

#### Overview
We're glad to announce that Spark OCR 3.12.0 has been released!
This release comes with new models for Handwritten Text Recognition, Spark 3.2 support, bug fixes, and notebook examples.

#### New Features

* Added to the ImageTextDetectorV2:
  * New parameter 'mergeIntersects': merge bounding boxes corresponding to detected text regions, when multiple bounding boxes that belong to the same text line overlap.
  * New parameter 'forceProcessing': now you can force processing of the results to avoid repeating the computation of results in pipelines where the same results are consumed by different transformers.
  * New feature: sizeThreshold parameter sets the expected size for the recognized text. From now on, text size will be automatically detected when sizeThreshold is set to -1.

* Added to the ImageToTextV2:
  * New parameter 'usePandasUdf': support PandasUdf to allow batch processing internally.
  * New support for formatted output, and HOCR. 
ocr.setOutputFormat(OcrOutputFormat.HOCR)
ocr.setOutputFormat(OcrOutputFormat.FORMATTED_TEXT)

* Support for Spark 3.2:
  * We added support for the latest Spark version, check installation instructions below.
  * Known problems & workarounds:
  
[SPARK-38330](https://issues.apache.org/jira/browse/SPARK-38330): S3 access issues, there's a workaround using the following settings,

```
//Scala
spark.sparkContext.hadoopConfiguration.set("fs.s3a.path.style.access", "true")

#Python
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
```

[SPARK-37577](https://issues.apache.org/jira/browse/SPARK-37577): changes in default behavior of query optimizer, it is already handled in start() function, or if you start the context manually, setting the following Spark properties,
```
#Python
spark.conf.set("spark.sql.optimizer.expression.nestedPruning.enabled", False)
spark.conf.set("spark.sql.optimizer.nestedSchemaPruning.enabled", False)
```

* Improved documentation on the website.

#### New Models

ocr_small_printed: Text recognition small model for printed text based on ImageToTextV2
ocr_small_handwritten: Text recognition small model for handwritten text based on ImageToTextV2
ocr_base_handwritten: Text recognition base model for handwritten text based on ImageToTextV2

#### Bug Fixes

* display_table() function failing to display tables coming from digital PDFs.

#### New notebooks

* [SparkOcrImageToTextV2OutputFormats.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3120-release-candidate/jupyter/TextRecognition/SparkOcrImageToTextV2OutputFormats.ipynb), different output formats for ImageToTextV2.

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_11_0">Version 3.11.0</a>
    </li>
    <li>
        <strong>Version 3.12.0</strong>
    </li>
    <li>
        <a href="release_notes_3_13_0">Version 3.13.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_3_14_0">3.14.0</a></li>
  <li><a href="release_notes_3_13_0">3.13.0</a></li>
  <li class="active"><a href="release_notes_3_12_0">3.12.0</a></li>
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