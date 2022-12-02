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

## 4.2.0

Release date: 31-10-2022


We are glad to announce that Spark OCR 4.2.0 has been released. This is mostly a compatibility release to ensure compatibility of Spark OCR against Spark NLP 4.2.1, and Spark NLP Healthcare 4.2.1.

#### Improvements
* Improved memory consumption and performance in the training of Visual NER models.

#### New Features
* PdfToForm new param: useFullyQualifiedName, added capability to return fully qualified key names.

#### New or Updated Notebooks
* [SparkOcrProcessMultiplepageScannedPDF.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrProcessMultiplepageScannedPDF.ipynb) has been added to show how to serve a multi-page document processing pipeline.
* [SparkOcrDigitalFormRecognition.ipynb](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/4.2.0-release-candidate/jupyter/FormRecognition/SparkOcrDigitalFormRecognition.ipynb) has been updated to show utilization of useFullyQualifiedName parameter.



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


## 4.0.2

Release date: 12-09-2022

#### Overview

We are glad to announce that Spark OCR 4.0.2 has been released!
This release comes with new features, fixes and more!.


#### New Features

* VisualDocumentClassifierV2 is now trainable! Continuing with the effort to make all the most useful models easily trainable, we added training capabilities to this annotator.
* Added support for Simplified Chinese.
* Added new 'PdfToForm' annotator, capable of extracting forms from digital PDFs. This is different from previously introduced VisualDocumentNER annotator in that this new annotator works only on digital documents, as opposite to the scanned forms handled by VisualDocumentNER. PdfToForm is complementary to VisualDocumentNER.
 

#### Improvements

* Support for multi-frame dicom has been added.
* Added the missing load()​ method in ImageToTextV2.

 

#### New Notebooks

* We added two new notebooks for VisualDocumentClassifierV2, a [preprocessing notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/VisualDocumentClassifierTraining/Spark-ocr%20visual%20doc%20classifier%20v2%20preprocessing%20on%20databricks.ipynb), useful when you're dealing with large datasets, and a [fine-tuning notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/VisualDocumentClassifierTraining/SparkOCRVisualDocumentClassifierv2Training.ipynb).
* We added a [new sample notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/FormRecognition/SparkOcrDigitalFormRecognition.ipynb) showing how to extract forms from digital PDF documents.
* We added a [new sample notebook](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/TextRecognition/SparkOcrImageToText-Chinese.ipynb) explaining how to use Simplified Chinese OCR.

 

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


<div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>

<ul class="pagination">
    <li>
        <a href="spark_ocr_versions/release_notes_3_13_0">Versions 3.13.0</a>
    </li>
    <li>
        <strong>Versions 3.14.0</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
  <li class="active"><a href="spark_ocr_versions/release_notes_3_14_0">3.14.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_13_0">3.13.0</a></li>
  <li><a href="spark_ocr_versions/release_notes_3_12_0">3.12.0</a></li>
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
