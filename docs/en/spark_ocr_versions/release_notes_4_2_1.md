---
layout: docs
header: true
seotitle: Visual NLP(Spark OCR)
title: Visual NLP(Spark OCR) release notes 4.2.1
permalink: /docs/en/spark_ocr_versions/release_notes_4_2_1
key: docs-release-notes
modify_date: "2022-11-28"
show_nav: true
sidebar:
    nav: spark-ocr
---

<div class="h3-box" markdown="1">

## 4.2.1

Release date: 11-28-2022

 
We're glad to announce that Spark-OCR 4.2.1 has been released! This release is almost completely about LightPipelines.
 

#### LightPipeline added to Spark-OCR
Originally introduced by Spark-NLP, this has been one of the most celebrated features by our users. In a nutshell, LightPipelines allow you switching your pipeline from distributed processing to local mode, in a single line of code. Also, results are much easier to post-process as they come in plain Python data structures. 

Now, LightPipelines are available in Spark-OCR as well! This is an initial implementation only covering three of our most popular annotators: ImageToText, PdfToImage, and BinaryToImage. Although not all the annotators from Spark-OCR are included in this initial release, a number of interesting features are being delivered:

* Latency has been dramatically reduced for small input dataset sizes.
* Interoperability with Spark-NLP and Spark-NLP healthcare: you can mix any NLP annotator with supported OCR annotators on the same LightPipeline.

Following is a chart comparing performance of different techniques on batches of different page counts: 8, 16, 24, 32, 40, 48, and 80 pages.

![image](/assets/images/ocr/light_pipelines.png)

For the 8 pages case, on the left side of the chart, LightPipelines average 1.25s per page vs. 4s per page that were scored by a similar Pytesseract implementation. That makes LightPipelines a great candidate to achieve low latency on small sized batches, while still leveraging parallelism.

#### Korean Support
You can start using Korean language by just passing the 'KOR' option to ImageToText,
```python
...
    # Run OCR
    ocr = ImageToText()
    # Set Korean language
    ocr.setLanguage(Language.KOR)
    # Download model from JSL S3
    ocr.setDownloadModelData(True)
```

#### Bug Fixes
* AlabReader has been updated to handle the new structure present in Annotation Lab's exported annotations.

#### New Notebooks
* Check how to use LightPipelines in this notebook: [SparkOcrLightPipelines.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/4.2.1-release-candidate/jupyter/SparkOcrLightPipelines.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}
