---
layout: docs
header: true
title: Spark OCR release notes
permalink: /docs/en/ocr_release_notes
key: docs-ocr-release-notes
modify_date: "2020-04-08"
show_nav: true
sidebar:
    nav: spark-ocr
---

## 3.8.0

Release date: 14-09-2021

#### Overview

Support Microsoft PPT and PPTX documents.

#### New Features

* Added [PptToPdf](ocr_pipeline_components#ppttopdf) transformer for convert PPT and PPTX slides to the PDF document.
* Added [PptToTextTable](ocr_pipeline_components#ppttotexttable) transformer for extract tables from PPT and PPTX slides.


#### New and updated notebooks

* [Convert PPT to PDF](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.8.0/jupyter/SparkOcrPptToPdf.ipynb) (New)
* [Extract tables from PPT](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.8.0/jupyter/SparkOcrPptToTextTable.ipynb) (New)


## 3.7.0

Release date: 30-08-2021

#### Overview

Improve table recognition and render OCR results to the PDF with original image


#### New Features

* Added [ImageToTextPdf](ocr_pipeline_components#imagetotextpdf) transformer for store recognized text to the searchable
PDF with original image
* Added [PdfAssembler](ocr_pipeline_components#pdfassembler) for assemble multipage PDF document from single page PDF
documents


#### Enhancements

* Added support dbfs for store models. This allow to use models on Databricks.
* Improved [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added params for tune [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector) algorithms
* Added possibility to render detected lines to the original image in [ImageTableCellDetector](ocr_table_recognition#imagetablecelldetector)
* Added support store recognized results to CSV in [ImageCellsToTextTable](ocr_table_recognition#imagecellstotexttable)
* Added [display_table](ocr_structures#displaytable) and [display_tables](ocr_structures#displaytables) functions
* Added [display_pdf_file](ocr_structures#displaypdffile) function for display pdf in embedded pdf viewer
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

Handwritten detection and improve visualization.


#### New Features

* Added [ImageHandwrittenDetector](ocr_object_detection#imagehandwrittendetector) for detect 'signature', 'date', 'name',
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

* [ImageSignatureDetector](ocr_object_detection#imagehandwrittendetector) is a DL model for detect signature on the image.


#### New notebooks

* [Image Signature Detection example](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/3.4.0/jupyter/SparkOcrImageSignatureDetection.ipynb)


## 3.3.0

Release date: 14-06-2021

#### Overview

Table detection and recognition for scanned documents.

For table detection we added ___ImageTableDetector___. 
It based on __CascadeTabNet__ which used _Cascade mask Region-based CNN High-Resolution Network_ (Cascade mask R-CNN HRNet).
Model was pre-trained on the __COCO dataset__ and fine tuned on __ICDAR 2019__ competitions dataset for table detection. It demonstrates state of the art results for ICDAR 2013 and TableBank. And top results for ICDAR 2019.

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

Image processing on GPU. It is in 3..5 times faster than on CPU.

More details please read in [GPU image preprocessing in Spark OCR](https://medium.com/spark-nlp/gpu-image-pre-processing-in-spark-ocr-3-1-0-6fc27560a9bb)


#### New Features

* [GPUImageTransformer](ocr_pipeline_components#gpuimagetransformer) with support: scaling, erosion, delation, Otsu and Huang thresholding.
* Added [display_images](ocr_structures#displayimages) util function for display images from Spark DataFrame in Jupyter notebooks.

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



## 1.11.0

Release date: 25-02-2021

#### Overview

Support German, French, Spanish and Russian languages.
Improving [PositionsFinder](ocr_pipeline_components#positionsfinder) and ImageToText for better support de-identification.

#### New Features

* Loading model data from S3 in [ImageToText](ocr_pipeline_components#imagetotext).
* Added support German, French, Spanish, Russian languages in [ImageToText](ocr_pipeline_components#imagetotext).
* Added different OCR model types: Base, Best, Fast in [ImageToText](ocr_pipeline_components#imagetotext).

#### Enhancements

* Added spaces symbols to the output positions in the [ImageToText](ocr_pipeline_components#imagetotext) transformer.
* Eliminate python-levensthein from dependencies for simplify installation.

#### Bugfixes

* Fixed issue with extracting coordinates in  in [ImageToText](ocr_pipeline_components#imagetotext).
* Fixed loading model data on cluster in yarn mode.

#### New notebooks

* [Languages Support](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/1.11.0/jupyter/SparkOcrLanguagesSupport.ipynb)
* [Image DeIdentification](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/1.11.0/jupyter/SparkOcrImageDeIdentification.ipynb)


## 1.10.0

Release date: 20-01-2021

#### Overview

Support Microsoft Docx documents.

#### New Features

* Added [DocToText](ocr_pipeline_components#doctotext) transformer for extract text
from DOCX documents.
* Added [DocToTextTable](ocr_pipeline_components#doctotexttable) transformer for extract
table data from DOCX documents.
* Added [DocToPdf](ocr_pipeline_components#doctopdf) transformer for convert DOCX
 documents to PDF format.

#### Bugfixes

* Fixed issue with loading model data on some cluster configurations



## 1.9.0

Release date: 11-12-2020

#### Overview

Extension of  FoundationOne report parser and support HOCR output format.

#### New Features

* Added [ImageToHocr](ocr_pipeline_components#imagetohocr) transformer for recognize text from image and store it to HOCR format.
* Added parsing gene lists from 'Appendix' in [FoundationOneReportParser](ocr_pipeline_components#foundationonereportparser) transformer.


## 1.8.0

Release date: 20-11-2020

#### Overview

Optimisation performance for processing multipage PDF documents.
Support up to 10k pages per document.

#### New Features

* Added [ImageAdaptiveBinarizer](ocr_pipeline_components#imageadaptivebinarizer) Scala transformer with support:
    - Gaussian local thresholding
    - Otsu thresholding
    - Sauvola local thresholding
* Added possibility to split pdf to small documents for optimize processing in [PdfToImage](ocr_pipeline_components#pdftoimage).


#### Enhancements

* Added applying binarization in [PdfToImage](ocr_pipeline_components#pdftoimage) for optimize memory usage.
* Added `pdfCoordinates` param to the [ImageToText](ocr_pipeline_components#imagetotext) transformer.
* Added 'total_pages' field to the [PdfToImage](ocr_pipeline_components#pdftoimage) transformer.
* Added different splitting strategies to the [PdfToImage](ocr_pipeline_components#pdftoimage) transformer.
* Simplified paging [PdfToImage](ocr_pipeline_components#pdftoimage) when run it with splitting to small PDF.
* Added params to the [PdfToText](ocr_pipeline_components#pdftotext) for disable extra functionality.
* Added `master_url` param to the python [start](ocr_install#using-start-function) function.


## 1.7.0

Release date: 22-09-2020

#### Overview

Support Spark 2.3.3.

#### Bugfixes

* Restored read JPEG2000 image


## 1.6.0

Release date: 05-09-2020

#### Overview

Support parsing data from tables for selectable PDFs.


#### New Features

* Added [PdfToTextTable](ocr_pipeline_components#pdftotexttable) transformer for extract tables from Pdf document per each page.
* Added [ImageCropper](ocr_pipeline_components#imagecropper) transformer for crop images.
* Added [ImageBrandsToText](ocr_pipeline_components#imagebrandstotext) transformer for detect text in defined areas.


## 1.5.0

Release date: 22-07-2020

#### Overview

FoundationOne report parsing support.

#### Enhancements

* Optimized memory usage during image processing


#### New Features

* Added [FoundationOneReportParser](ocr_pipeline_components#foundationonereportparser) which support parsing patient info,
genomic and biomarker findings.


## 1.4.0

Release date: 23-06-2020

#### Overview

Added support Dicom format and improved support image morphological operations.

#### Enhancements

* Updated [start](ocr_install#using-start-function) function. Improved support Spark NLP internal.
* `ImageMorphologyOpening` and `ImageErosion` are removed.
* Improved existing transformers for support de-identification Dicom documents.
* Added possibility to draw filled rectangles to [ImageDrawRegions](ocr_pipeline_components#imagedrawregions).

#### New Features

* Support reading and writing Dicom documents.
* Added [ImageMorphologyOperation](ocr_pipeline_components#imagemorphologyoperation) transformer which support:
 erosion, dilation, opening and closing operations.
 
#### Bugfixes

* Fixed issue in [ImageToText](ocr_pipeline_components#imagetotext) related to extraction coordinates.


## 1.3.0

Release date: 22-05-2020

#### Overview

New functionality for de-identification problem.

#### Enhancements

* Renamed TesseractOCR to ImageToText. 
* Simplified installation.
* Added check license from `SPARK_NLP_LICENSE` env varibale.

#### New Features

* Support storing for binaryFormat. Added support storing Image and PDF files.
* Support selectable pdf for [TextToPdf](ocr_pipeline_components#texttopdf) transformer.
* Added [UpdateTextPosition](ocr_pipeline_components#updatetextposition) transformer.


## 1.2.0

Release date: 08-04-2020


#### Overview

Improved support Databricks and processing selectable pdfs.

#### Enhancements

* Adapted Spark OCR for run on Databricks.
* Added rewriting positions in [ImageToText](ocr_pipeline_components#imagetotext) when run together with PdfToText.
* Added 'positionsCol' param to [ImageToText](ocr_pipeline_components#imagetotext).
* Improved support Spark NLP. Changed [start](/ocr_install#using-start-function) function.

#### New Features

* Added [showImage](ocr_structures#showimages) implicit to Dataframe for display images in Scala Databricks notebooks.
* Added [display_images](ocr_structures#display_images) function for display images in Python Databricks notebooks.
* Added propagation selectable pdf file in [TextToPdf](ocr_pipeline_components#texttopdf). Added 'inputContent' param to 'TextToPdf'.


## 1.1.2

Release date: 09-03-2020

#### Overview

Minor improvements and fixes

#### Enhancements

* Improved messages during license validation

#### Bugfixes

* Fixed dependencies issue


## 1.1.1

Release date: 06-03-2020

#### Overview

Integration with license server.

#### Enhancements

* Added license validation. License can be set in following waysq:
  - Environment variable. Set variable 'JSL_OCR_LICENSE'.
  - System property. Set property 'jsl.sparkocr.settings.license'.
  - Application.conf file. Set property 'jsl.sparkocr.settings.license'.
* Added auto renew license using jsl license server.


## 1.1.0

Release date: 03-03-2020

#### Overview

This release contains improvements for preprocessing image before run OCR and
added possibility to store results to PDF for keep original formatting.


#### New Features

* Added auto calculation maximum size of objects for removing in `ImageRemoveObjects`.
  This improvement avoids to remove `.` and affect symbols with dots (`i`, `!`, `?`).
  Added `minSizeFont` param to `ImageRemoveObjects` transformer for
  activate this functional.
* Added `ocrParams` parameter to `ImageToText` transformer for set any
  ocr params.
* Added extraction font size in `ImageToText`
* Added `TextToPdf` transformer for render text with positions to pdf file.


#### Enhancements

* Added setting resolution in `ImageToText`. And added `ignoreResolution` param with
  default `true` value to `ImageToText` transformer for back compatibility.
* Added parsing resolution from image metadata in `BinaryToImage` transformer.
* Added storing resolution in `PrfToImage` transformer.
* Added resolution field to Image schema.
* Updated 'start' function for set 'PYSPARK_PYTHON' env variable.
* Improve auto-scaling/skew correction:
   - improved access to images values
   - removing unnecessary copies of images
   - adding more test cases
   - improving auto-correlation in auto-scaling.


## 1.0.0

Release date: 12-02-2020

#### Overview

Spark NLP OCR functionality was reimplemented as set of Spark ML transformers and
moved to separate Spark OCR library.


#### New Features

* Added extraction coordinates of each symbol in ImageToText
* Added ImageDrawRegions transformer
* Added ImageToPdf transformer
* Added ImageMorphologyOpening transformer
* Added ImageRemoveObjects transformer
* Added ImageAdaptiveThresholding transformer


#### Enhancements

* Reimplement main functionality as Spark ML transformers
* Moved DrawRectangle functionality to PdfDrawRegions transformer
* Added 'start' function with support SparkMonitor initialization
* Moved PositionFinder to Spark OCR


#### Bugfixes

* Fixed bug with transforming complex pdf to image