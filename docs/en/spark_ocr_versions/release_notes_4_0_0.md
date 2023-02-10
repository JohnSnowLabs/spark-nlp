---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 4.0.0
permalink: /docs/en/spark_ocr_versions/release_notes_4_0_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 4.0.0

Release date: 16-07-2022

#### Overview

We are very glad to announce that Spark OCR 4.0.0 has been released!
This release comes with new models, new functionality, bug fixes, and compatibility with 4.0.0 versions of Spark NLP and Spark NLP for Healthcare.

#### New Features
* New DicomMetadataDeidentifier class to help deidentifying metadata of dicom files. Example Notebook.
* New helper function display_dicom() to help displaying DICOM files in notebooks.
* New DicomDrawRegions that can clean burned pixels for removing PHI.
* Improved support for DICOM files containing 12bit images.

#### Bug Fixes
* Fixes on the Visual NER Finetuning process including VisualDocumentNERv2 and AlabReader.
* Improved exception handling for VisualDocumentClassifier models.

#### New Models
* New LayoutLMv3 based Visual Document NER: layoutlmv3_finetuned_funsd.
* Improved handwritten detection ocr_base_handwritten_v2.
* VisualDocumentClassifierV2: layoutlmv2_rvl_cdip_40k. This model adds more data compared to layoutlmv2_rvl_cdip_1500, and achieves an accuracy of 88%.

#### Compatibility Updates
* Deprecated Spark 2.3 and Spark 2.4 support.
* Tested compatibility with Spark-NLP and Spark NLP for Healthcare 4.0.0.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}