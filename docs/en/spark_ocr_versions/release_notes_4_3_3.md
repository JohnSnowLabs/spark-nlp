---
layout: docs
header: true
seotitle: Spark OCR | John Snow Labs
title: Spark OCR release notes
permalink: /docs/en/spark_ocr_versions/release_notes_4_3_3
key: docs-ocr-release-notes
modify_date: "2023-03-14"
show_nav: true
sidebar:
    nav: spark-ocr
---

<div class="h3-box" markdown="1">

## 4.3.3

Release date: 14-03-2023

We're glad to announce that Visual NLP 😎 4.3.3 has been released.

### Highlights
* New parameter keepOriginalEncoding in PdfToHocr.
* New Yolo-based table and form detector. 
* Memory consumption in VisualQuestionAnswering and ImageTableDetector models has been improved.
* Fixes in AlabReader
* Fixes in HocrToTextTable.

#### New parameter keepOriginalEncoding in PdfToHocr
Now you can choose to make PdfToHocr return an ASCII normalized version of the characters present in the PDF(keepOriginalEncoding=False) or to return the original Unicode character(keepOriginalEncoding=True).
Source PDF,
![image](/assets/images/ocr/source.png)

Keeping the encoding,
![image](/assets/images/ocr/keeping.png)

Not keeping it,
![image](/assets/images/ocr/notkeeping.png)


#### New Yolo-based Table and Form detector
This new model allows to distinguish between forms and tables, so you can apply different downstream processing afterwards.

![image](/assets/images/ocr/form_tables.jpg)

Check a full example of utilization in [this notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImageTableAndFormDetection.ipynb).


#### Memory consumption in VisualQuestionAnswering and ImageTableDetector models has been improved
Memory utilization has been improved to make it more GC friendly. The practical result is that big jobs are more stable, and less likely to get restarted because of exhausting resources.


#### Fixes in AlabReader
AlabReader has been improved to fix some bugs, and to improve the performance.

#### Fixes in HocrToTextTable
HocrToTextTable has been improved in order to better handle some corner cases in which the last rows of tables were being missed.

This release of Visual NLP is compatible with version 4.3.1 of Spark-NLP and version 4.3.1 of Spark NLP for Healthcare.


</div><div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>

{%- include docs-sparckocr-pagination.html -%}
