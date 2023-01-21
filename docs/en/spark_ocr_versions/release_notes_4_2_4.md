---
layout: docs
header: true
seotitle: Visual NLP(Spark OCR)
title: Visual NLP(Spark OCR) release notes 4.2.4
permalink: /docs/en/spark_ocr_versions/release_notes_4_2_4
key: docs-release-notes
modify_date: "2022-12-19"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 4.2.4

We are glad to announce that Spark OCR 4.2.4⚡has been released!! 
This release includes new optimized ImageToTextV2 models, more support on annotators in LightPipelines, a new PdfToHocr annotator, enhancements, and more!

#### New Features
* New annotators supported in LightPipelines: PdfToText and most Image transformations. Check sample notebook for details.
* Handling of PDFs with broken headers: some PDFs may contain incorrect header information causing the pipelines to fail to process them, now PDF processing annotators support handling these documents.

#### New Annotators
* New ImageToTextV2 Transformers based OCR annotator, 
  * Intended to become a full replacement of original ImageToTextV2.
  * Speed ups of up to 2x compared to original model.
  * It doesn't require GPU, it works with CPU only environments.
  * Preliminary experiments show similar character error rate compared to original model.
  * Optimized versions take less space(about a half) and are faster to store and download.
  * Full JVM implementation.
  * Limitations: currently the new ImageToTextV2 doesn't support Hocr output.
  * To start using it, follow this example,

```python
...
from sparkocr.optimized import ImageToTextV2
ocr = ImageToTextV2.pretrained("ocr_base_printed_v2_opt", "en", "clinical/ocr")
```

* New PdfToHocr: this new annotator allows to produce HOCR output from digital PDFs. This is not only useful for integrating into existing annotators that already consume HOCR, but for new pipelines that will be released in the future. Stay tuned for new releases.  


#### New Models
ocr_base_printed_v2
ocr_base_handwritten_v2
ocr_base_printed_v2_opt (quantized version)
ocr_base_handwritten_v2_opt (quantized version)


#### New Notebooks
* New supported transformers in LightPipelines in action,
[SparkOcrLightPipelinesPdf.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/4.2.2-release-candidate/jupyter/SparkOcrLightPipelinesPdf.ipynb)

* PdfToHocr,
[SparkOCRPdfToHocr.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/4.2.2-release-candidate/jupyter/SparkOCRPdfToHocr.ipynb)

This release is compatible with Spark NLP 4.2.4, and Spark NLP for Healthcare 4.2.3.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}
