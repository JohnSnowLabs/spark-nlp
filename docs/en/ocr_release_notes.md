---
layout: article
title: Spark OCR release notes
permalink: /docs/en/ocr_release_notes
key: docs-ocr-release-notes
modify_date: "2020-02-27"
---

# 1.1.0

## Overview

This release contains improvements for preprocessing image before run OCR and
added possibility to store results to PDF for keep original formatting.


## New Features

* Added auto calculation maximum size of objects for removing in `ImageRemoveObjects`.
  This improvement avoids to remove `.` and affect symbols with dots (`i`, `!`, `?`).
  Added `minSizeFont` param to `ImageRemoveObjects` transformer for
  activate this functional.
* Added `tesseractParams` parameter to `TesseractOcr` transformer for set any
  tesseract params.
* Added extraction font size in `TesseractOcr`
* Added `TextToPdf` transformer for render text with positions to pdf file.


## Enhancements

* Added setting resolution in `TesseractOcr`. And added `ignoreResolution` param with
  default `true` value to `TesseractOcr` transformer for back compatibility.
* Added parsing resolution from image metadata in `BinaryToImage` transformer.
* Added storing resolution in `PrfToImage` transformer.
* Added resolution field to Image schema.
* Updated 'start' function for set 'PYSPARK_PYTHON' env variable.
* Improve auto-scaling/skew correction:
   - improved access to images values
   - removing unnecessary copies of images
   - adding more test cases
   - improving auto-correlation in auto-scaling.


# 1.0.0

Release date: 12-02-2020

## Overview

Spark NLP OCR functionality was reimplemented as set of Spark ML transformers and
moved to separate Spark OCR library.


## New Features

* Added extraction coordinates of each symbol in TesseractOCR
* Added ImageDrawRegions transformer
* Added ImageToPdf transformer
* Added ImageMorphologyOpening transformer
* Added ImageRemoveObjects transformer
* Added ImageAdaptiveThresholding transformer


## Enhancements

* Reimplement main functionality as Spark ML transformers
* Moved DrawRectangle functionality to PdfDrawRegions transformer
* Added 'start' function with support SparkMonitor initialization
* Moved PositionFinder to Spark OCR


## Bugfixes

* Fixed bug with transforming complex pdf to image