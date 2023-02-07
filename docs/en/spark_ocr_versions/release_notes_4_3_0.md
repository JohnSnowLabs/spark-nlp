---
layout: docs
header: true
seotitle: Visual NLP(Spark OCR)
title: Visual NLP(Spark OCR) release notes 4.3.0
permalink: /docs/en/spark_ocr_versions/release_notes_4_3_0
key: docs-ocr-release-notes
modify_date: "2023-01-13"
show_nav: true
sidebar:
    nav: spark-ocr
---

<div class="h3-box" markdown="1">

## 4.3.0

Release date: 2023-01-13

We are glad to announce that Spark OCR 4.3.0 has been released!! This big release comes with improvements in Dicom Processing, Visual Question Answering, new Table Extraction annotators, and much more!.

 
### New Features
* PositionFinder now works in LightPipelines.
* New annotator HocrToTextTable to work together with PdfToHocr that allows table extraction from digital PDFs. This allows to extract tables using a mixed pipeline in which tables are detected using visual features, but the text is pulled directly from the digital layer of the PDF yielding near to perfect results, and removing OCR overhead.

* New Dicom Processing improvements,

  * Added support of Dicom documents to BinaryFile Datasource: this allows to write Dicom documents from Spark Dataframes to all data storages supported by Spark, in batch and streaming mode.
  * Added possibility to specify name of the files in BinaryFile Datasource: now we can store images, PDFs, Dicom files directly using Spark capabilities with names of our choice, overcoming the limitation imposed by Spark of naming files according to partitions.
  * Added DicomToMetadata Transformer: it allows to extract metadata from the Dicom documents. This allows to analyze Dicom metadata using Spark capabilities. For example, collect statistic about color schema, number of frames, compression of the images. This is useful for estimating needed resources and time before starting to process a big dataset.
  * Added DicomToImageV3 based on Pydicom with better support of different color schemas. Added support YBR_FULL_422, YBR_FULL images. Also fixed handling pixel data with different pixel size for RGB and Monochrome images.
  * Added support for compression after update pixel data in DicomDrawRegions. This reduces size of output Dicom files by applying JPEGBaseline8Bit compression to the pixel data.
  * Added support for different color schemas in DicomDrawRegions. Added support YBR_FULL_422, YBR_FULL images.
  * Added support for coordinates with rotated bounding box in DicomDrawRegions for compatibility with ImageTextDetectorV2.
* Fixed ImageTextDetectorV2 for images without text.
* New Donut based VisualQuestionAnswering annotator. 

Supports two modes of operation: it can receive an array of questions in the same row as the input image; in this way, each input image can be queried by an arbitrary set of user-defined questions, and also questions can be defined globally outside the Dataframe. This will cause that all images will be queried by the same set of questions.
Running time is about a half the time per question when compared to the open-source version.
Optimized model is smaller(about a half) of the original open-source version, making it easier to download and distribute in a cluster.
Two models available: `docvqa_donut_base` and `docvqa_donut_base_opt`(quantized).
LightPipelines support.
 	

#### Bug Fixes

* Empty tables now handled properly in ImageCellsToTextTable.
* Pretrained models for VisualDocumentNerV21 are now accessible.

#### New/updated Notebooks

* [SparkOcrVisualQuestionAnswering.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrVisualQuestionAnswering.ipynb), this notebook shows examples on how to use Donut based visual question answering in Spark-OCR.

* [SparkOCRPdfToTable.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRPdfToTable.ipynb), this notebook shows how PdfToHocr and HocrToTextTable can be put together to do table extraction without OCR, by just relying on the digital layer of text in the PDF. Still, existent well tested table detection models, continue to be used for finding the tables.

* [SparkOcrImageTableRecognitionWHOCR.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImageTableRecognitionWHOCR.ipynb), this notebook shows table detection, and the HocrToTextTable in action. Compared to previous implementations, now the OCR method is external, and it can be replaced by different implementations(even handwritten!).

This release is compatible with Spark NLP for Healthcare 4.2.4, and Spark NLP 4.2.4.

</div><div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>

{%- include docs-sparckocr-pagination.html -%}
