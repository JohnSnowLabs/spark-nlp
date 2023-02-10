---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 4.1.0
permalink: /docs/en/spark_ocr_versions/release_notes_4_1_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 4.1.0

Release date: 22-09-2022

#### Overview

We are glad to announce that Spark OCR 4.1.0 has been released!
This release comes with new features, enhancements, fixes and more!.
 

#### New Features
* DicomSplitter: new annotator that helps to distribute and split Dicom files into multiple frames. It supports multiple strategies, similar to our PdfToImage annotator. It enables parallel processing of different frames and keeps memory utilization bounded. For big datasets, or memory constrained environments, it enables Streaming Mode to process frames 1-by-1, resulting in very low memory requirements.

* DicomToImageV2: new annotator that supports loading images from Dicom files/frames, without loading Dicom files into memory. Targeted to datasets containing big Dicom files.
* This is an example on how to use the two above mentioned annotators to process images, coming from your big Dicom files in a memory constrained setting,

```
        splitter = DicomSplitter()


        splitter.setInputCol("path")
        splitter.setOutputCol("frames")
        splitter.setSplitNumBatch(2)
        splitter.setPartitionNum(2)

        dicom = DicomToImageV2()
        dicom.setInputCols(["path", "frames"])
        dicom.setOutputCol("image")

        pipeline = PipelineModel(stages=[
            splitter,
            dicom
        ])
```


* New image pre-processing annotators: ImageHomogenizeLight, ImageRemoveBackground, ImageEnhanceContrast, ImageRemoveGlare. For examples on how to use them, and their amazing results check this notebook: [SparkOcrImagePreprocessing.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImagePreprocessing.ipynb).
 

#### Improvements
* VisualDocumentClassifierV2 training has been improved for more efficient memory utilization.
* Library dependencies have been updated to remove security vulnerabilities.


#### Bug Fixes
* The infamous "ImportError: No module named resource" bug that was affecting Windows users has been fixed.
* Some issues while loading images using AlabReader have been fixed.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}