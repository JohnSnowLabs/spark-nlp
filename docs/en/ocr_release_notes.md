---
layout: docs
header: true
seotitle: Spark OCR | John Snow Labs
title: Spark OCR release notes
permalink: /docs/en/ocr_release_notes
key: docs-ocr-release-notes
modify_date: "2020-04-08"
show_nav: true
sidebar:
    nav: spark-ocr
---


## 3.14.0

Release date: 13-06-2022

#### Overview

We are glad to announce that Spark OCR 3.14.0 has been released!.
This release focuses around Visual Document Classification models, native Image Preprocessing on the JVM, and bug fixes.

#### New Features

* VisualDocumentClassifierv2:
  * New annotator for classifying documents based on multimodal(text + images) features.
  
* VisualDocumentClassifierv3: 
  * New annotator for classifying documents based on image features.
 
* ImageTransformer:
  * New transformer that provides different image transformations on the JVM. Supported transforms are Scaling, Adaptive Thresholding, Median Blur, Dilation, Erosion, and Object Removal.


#### New notebooks

+ [SparkOCRVisualDocumentClassifierv2.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRVisualDocumentClassifierv2.ipynb), example of Visual Document Classification using multimodal (text + visual) features.
+ [SparkOCRVisualDocumentClassifierv3.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRVisualDocumentClassifierv3.ipynb), example of Visual Document Classification using only visual features.
+ [SparkOCRCPUImageOperations.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.14.0-release-candidate/jupyter/SparkOCRCPUImageOperations.ipynb), example of ImageTransformer.



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

## Previos versions

</div>

<ul class="pagination">
    <li>
        <a href="spark_ocr_versions/release_notes_3_11_0">Versions 3.11.0</a>
    </li>
    <li>
        <strong>Versions 3.12.0</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
  <li class="active"><a href="spark_ocr_versions/release_notes_3_12_0">3.12.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_11_0">3.11.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_10_0">3.10.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_9_1">3.9.1</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_9_0">3.9.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_8_0">3.8.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_7_0">3.7.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_6_0">3.6.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_5_0">3.5.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_4_0">3.4.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_3_0">3.3.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_2_0">3.2.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_1_0">3.1.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_0_0">3.0.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_11_0">1.11.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_10_0">1.10.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_9_0">1.9.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_8_0">1.8.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_7_0">1.7.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_6_0">1.6.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_5_0">1.5.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_4_0">1.4.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_3_0">1.3.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_2_0">1.2.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_2">1.1.2</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_1">1.1.1</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_1_0">1.1.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_1_0_0">1.0.0</a></li>
</ul>