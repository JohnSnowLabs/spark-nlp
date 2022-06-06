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


## 3.11.0

Release date: 28-02-2022


#### Overview

We are glad to announce that Spark OCR 3.11.0 has been released!.
This release comes with new models, new features, bug fixes, and notebook examples.

#### New Features

* Added [ImageTextDetectorV2](ocr_object_detection#imagetextdetectorv2) Python Spark-OCR Transformer for detecting printed and handwritten text
 using CRAFT architecture with Refiner Net.
* Added [ImageTextRecognizerV2](ocr_pipeline_components#imagetotextv2) Python Spark-OCR Transformer for recognizing
 printed and handwritten text based on Deep Learning Transformer Architecture.
* Added [FormRelationExtractor](ocr_visual_document_understanding#formrelationextractor) for detecting relations between key and value entities in forms.
* Added the capability of fine tuning VisualDocumentNerV2 models for key-value pairs extraction.

#### New Models

* ImageTextDetectorV2: this extends the ImageTextDetectorV1 character level text detection model with a refiner net architecture.
* ImageTextRecognizerV2: Text recognition for printed text based on the Deep Learning Transformer Architecture.

#### New notebooks

* [SparkOcrImageToTextV2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextRecognition/SparkOcrImageToTextV2.ipynb)
* [ImageTextDetectorV2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextDetection/SparkOcrImageTextDetectionV2.ipynb)
* [Visual Document NER v2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/TextRecognition/SparkOcrImageToTextV2.ipynb)
* [SparkOcrFormRecognition](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/FormRecognition/SparkOcrFormRecognition.ipynb)
* [SparkOCRVisualDocumentNERv2FineTune](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOCRVisualDocumentNERv2FineTune.ipynb)
* Creating Rest a API with Synapse to extract text from images, [SparkOcrRestApi](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOcrRestApi.ipynb)
* Creating Rest a API with Synapse to extract text from PDFs, [SparkOcrRestApiPdf](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3110-release-candidate/jupyter/SparkOcrRestApiPdf.ipynb)

## 3.10.0

Release date: 10-01-2022


#### Overview

Form recognition using LayoutLMv2 and text detection.


#### New Features

* Added [VisualDocumentNERv2](ocr_visual_document_understanding#visualdocumentnerv2) transformer
* Added DL based [ImageTextDetector](ocr_object_detection#imagetextdetector) transformer
* Support rotated regions in [ImageSplitRegions](ocr_pipeline_components#imagesplitregions)
* Support rotated regions in [ImageDrawRegions](ocr_pipeline_components#imagedrawregions)


#### New Models

* LayoutLMv2 fine-tuned on FUNSD dataset
* Text detection model based on CRAFT architecture


#### New notebooks

* [Text Detection](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3100-release-candidate/jupyter/TextDetection/SparkOcrImageTextDetection.ipynb)
* [Visual Document NER v2](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3100-release-candidate/jupyter/SparkOCRVisualDocumentNERv2.ipynb)



## 3.9.1

Release date: 02-11-2021

#### Overview

Added preserving of original file formatting

#### Enhancements

* Added keepLayout param to the [ImageToText](ocr_pipeline_components#imagetotext)

#### New and updated notebooks

* [Preserve Original Formatting](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.9.1/jupyter/SparkOcrPreserveOriginalFormatting.ipynb)



## 3.9.0

Release date: 20-10-2021

#### Overview

Improve visualization and support Spark NLP. 

#### New Features

* Added [HocrTokenizer](ocr_pipeline_components#hocrtokenizer)
* Added [HocrDocumentAssembler](ocr_pipeline_components#hocrdocumentassembler)
* Added [ImageDrawAnnotations](ocr_pipeline_components#imagedrawannotations)
* Added support Arabic language in ImageToText and ImageToHocr

#### Enhancements

* Added postprocessing to the [ImageTableDetector](ocr_table_recognition#imagetabledetector)
* Added Spark NLP by default to spark session in start function
* Changed default value for ignoreResolution param in [ImageToText](ocr_pipeline_components#imagetotext)
* Updated license-validator. Added support floating license and set AWS keys from license.
* Added 'whiteList' param to the [VisualDocumentNER](ocr_pipeline_components#visualdocumentner)

#### New and updated notebooks

* [Spark OCR HOCR](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.9.0/jupyter/SparkOcrHocr.ipynb)
* [Visual Document NER](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.9.0/jupyter/SparkOCRVisualDocumentNer.ipynb)


## 3.8.0

Release date: 14-09-2021

#### Overview

Support Microsoft PPT and PPTX documents.

#### New Features

* Added [PptToPdf](ocr_pipeline_components#ppttopdf) transformer for converting PPT and PPTX slides to the PDF document.
* Added [PptToTextTable](ocr_pipeline_components#ppttotexttable) transformer for extracting tables from PPT and PPTX slides.


#### New and updated notebooks

* [Convert PPT to PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.8.0/jupyter/SparkOcrPptToPdf.ipynb) (New)
* [Extract tables from PPT](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.8.0/jupyter/SparkOcrPptToTextTable.ipynb) (New)


## 3.7.0

Release date: 30-08-2021

#### Overview

Improve table recognition and render OCR results to the PDF with original image


#### New Features

* Added [ImageToTextPdf](ocr_pipeline_components#imagetotextpdf) transformer for storing recognized text to the searchable
PDF with original image
* Added [PdfAssembler](ocr_pipeline_components#pdfassembler) for assembling multipage PDF document from single page PDF
documents


#### Enhancements

* Added support dbfs for store models. This allow to use models on Databricks.
* Improved [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added params for tuning [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added possibility to render detected lines to the original image in [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector)
* Added support to store recognized results to CSV in [ImageCellsToTextTable](ocr_table_recognition#imagecellstotexttable)
* Added [display_table](ocr_structures#displaytable) and [display_tables](ocr_structures#displaytables) functions
* Added [display_pdf_file](ocr_structures#displaypdffile) function for displaying pdf in embedded pdf viewer
* Updated license validator


#### New and updated notebooks

* [Process multiple page scanned PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrProcessMultiplepageScannedPDF.ipynb) (New)
* [Image Table Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableDetection.ipynb)
* [Image Cell Recognition example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableCellRecognition.ipynb)
* [Image Table Recognition](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableRecognition.ipynb)
* [Tables Recognition from PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.7.0/jupyter/SparkOcrImageTableRecognitionPdf.ipynb)



## 3.6.0

Release date: 05-08-2021

#### Overview

Handwritten detection and visualization improvement.


#### New Features

* Added [ImageHandwrittenDetector](ocr_object_detection#imagehandwrittendetector) for detecting 'signature', 'date', 'name',
 'title', 'address' and others handwritten text.
* Added rendering labels and scores in [ImageDrawRegions](ocr_pipeline_components#imagedrawregions).
* Added possibility to scale image to fixed size in [ImageScaler](ocr_pipeline_components#imagescaler)
 with keeping original ratio.


#### Enhancements

* Support new version of pip for installing python package
* Added support string labels for detectors
* Added an auto inferencing of the input shape for detector models
* New license validator


#### Bugfixes

* Fixed display BGR images in display functions


#### New and updated notebooks

* [Image Signature Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageSignatureDetection.ipynb)
* [Image Handwritten Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageHandwrittenDetection.ipynb)
* [Image Scaler example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageScaler.ipynb)


## 3.5.0

Release date: 15-07-2021

#### Overview

Improve table detection and table recognition.

More details please read in [Extract Tabular Data from PDF in Spark OCR](https://medium.com/spark-nlp/extract-tabular-data-from-pdf-in-spark-ocr-b02136bc0fcb)


#### New Features

* Added new method to [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) which support 
borderless tables and combined tables.
* Added __Wolf__ and __Singh__ adaptive binarization methods to the [ImageAdaptiveThresholding](ocr_pipeline_components#imageadaptivethresholding).


#### Enhancements

* Added possibility to use different type of images as input for [ImageTableDetector](ocr_table_recognition#imagetabledetector).
* Added [display_pdf](ocr_structures#displaypdf) and [display_images_horizontal](ocr_structures#displayimageshorizontal) util functions.

#### New notebooks

* [Tables Recognition from PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.5.0/jupyter/SparkOcrImageTableRecognitionPdf.ipynb)
* [Pdf de-identification on Databricks](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.5.0/databricks/python/SparkOcrDeIdentification.ipynb)
* [Dicom de-identification on Databricks](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.5.0/databricks/python/SparkOCRDeIdentificationDicom.ipynb)


## 3.4.0

Release date: 30-06-2021

#### Overview

Signature Detection in image-based documents.

More details please read in [Signature Detection in Spark OCR](https://medium.com/spark-nlp/signature-detection-in-spark-ocr-32f9e6f91e3c)

#### New Features

* [ImageSignatureDetector](ocr_object_detection#imagehandwrittendetector) is a DL model for detecting signature on the image.


#### New notebooks

* [Image Signature Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.4.0/jupyter/SparkOcrImageSignatureDetection.ipynb)


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


## 3.2.0

Release date: 28-05-2021

#### Overview

Multi-modal visual document understanding, built on the LayoutLM architecture.
It achieves new state-of-the-art accuracy in several downstream tasks,
including form understanding and receipt understanding.


#### New Features

* [VisualDocumentNER](ocr_pipeline_components#visualdocumentner) is a DL model for NER problem using text and layout data.
  Currently available pre-trained model on the SROIE dataset.


#### Enhancements

* Added support `SPARK_OCR_LICENSE` env key for read license.
* Update dependencies and sync Spark versions with Spark NLP.


#### Bugfixes

* Fixed an issue that some ImageReaderSpi plugins are unavailable in the fat jar.

#### New notebooks

* [Visual Document NER](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.2.0/jupyter/SparkOCRVisualDocumentNer.ipynb)



## 3.1.0

Release date: 16-04-2021

#### Overview

Image processing on GPU. It is in 3.5 times faster than on CPU.

More details please read in [GPU image preprocessing in Spark OCR](https://medium.com/spark-nlp/gpu-image-pre-processing-in-spark-ocr-3-1-0-6fc27560a9bb)


#### New Features

* [GPUImageTransformer](ocr_pipeline_components#gpuimagetransformer) with support: scaling, erosion, delation, Otsu and Huang thresholding.
* Added [display_images](ocr_structures#displayimages) util function for displaying images from Spark DataFrame in Jupyter notebooks.

#### Enhancements

* Improve [display_image](ocr_structures#displayimage) util function.

#### Bug fixes

* Fixed issue with extra dependencies in [start](ocr_install#using-start-function) function

#### New notebooks

* [GPU image processing](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.1.0/jupyter/SparkOCRGPUOperations.ipynb)



## 3.0.0

Release date: 02-04-2021

#### Overview

We are very excited to release Spark OCR 3.0.0!

Spark OCR 3.0.0 extends the support for Apache Spark 3.0.x and 3.1.x major releases on Scala 2.12 with both Hadoop 2.7. and 3.2. We will support all 4 major Apache Spark and PySpark releases of 2.3.x, 2.4.x, 3.0.x, and 3.1.x.

Spark OCR started to support Tensorflow models. First model is [VisualDocumentClassifier](ocr_pipeline_components#visualdocumentclassifier).

#### New Features

* Support for Apache Spark and PySpark 3.0.x on Scala 2.12
* Support for Apache Spark and PySpark 3.1.x on Scala 2.12
* Support 9x new Databricks runtimes:
  * Databricks 7.3
  * Databricks 7.3 ML GPU
  * Databricks 7.4
  * Databricks 7.4 ML GPU
  * Databricks 7.5
  * Databricks 7.5 ML GPU
  * Databricks 7.6
  * Databricks 7.6 ML GPU
  * Databricks 8.0
  * Databricks 8.0 ML (there is no GPU in 8.0)
  * Databricks 8.1
* Support 2x new EMR 6.x: 
  * EMR 6.1.0 (Apache Spark 3.0.0 / Hadoop 3.2.1)
  * EMR 6.2.0 (Apache Spark 3.0.1 / Hadoop 3.2.1)
* [VisualDocumentClassifier](ocr_pipeline_components#visualdocumentclassifier) model for classification documents using text and layout data.
* Added support Vietnamese language.

#### New notebooks

* [Visual Document Classifier](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRVisualDocumentClassifier.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Previos versions

</div>

<ul class="pagination">
    <li>
        <a href="ocr_release_notes_2">Versions 1.0.0</a>
    </li>
    <li>
        <strong>Versions 3.0.0</strong>
    </li>
</ul>