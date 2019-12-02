---
layout: article
title: Spark OCR (Licensed)
permalink: /docs/en/ocr
key: docs-ocr
modify_date: "2019-09-06"
---
Spark NLP comes with an OCR module that can read both PDF files and scanned images (requires `Tesseract 4.x+`).

## Installation

### Installing Tesseract

As mentioned above, if you are dealing with scanned images instead of test-selectable PDF files you need to install `tesseract 4.x+` on all the nodes in your cluster. Here how you can install it on Ubuntu/Debian:

```bash
apt-get install tesseract-ocr
```

In `Databricks` this command may result in installing `tesseract 3.x` instead of version `4.x`.

You can simply run this `init script` to install `tesseract 4.x` in your Databricks cluster:

```bash
#!/bin/bash
sudo apt-get install -y g++ # or clang++ (presumably)
sudo apt-get install -y autoconf automake libtool
sudo apt-get install -y pkg-config
sudo apt-get install -y libpng-dev
sudo apt-get install -y libjpeg8-dev
sudo apt-get install -y libtiff5-dev
sudo apt-get install -y zlib1g-dev
​
wget http://www.leptonica.org/source/leptonica-1.74.4.tar.gz
tar xvf leptonica-1.74.4.tar.gz
cd leptonica-1.74.4
./configure
make
sudo make install
​
git clone --single-branch --branch 4.1 https://github.com/tesseract-ocr/tesseract.git
cd tesseract
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
​
tesseract -v
```

## Quick start

Let's read a PDF file:

```scala
import com.johnsnowlabs.nlp.util.io.OcrHelper
val ocrHelper = new OcrHelper()

//If you do this locally you can use file:/// or hdfs:/// if the files are hosted in Hadoop
val dataset = ocrHelper.createDataset(spark, "/tmp/sample_article.pdf")

```

If you are trying to extract text from scanned images in the format of PDF, please keep in mind to use these configs:

```scala
ocrHelper.setPreferredMethod("image")
ocrHelper.setFallbackMethod(false)
ocrHelper.setMinSizeBeforeFallback(0)
```

## Configuration

* `setPreferredMethod(text/image = text)` either `text` or `image` will work. Defaults to `text`. Text mode works better and faster for digital or text scanned PDFs
* `setFallbackMethod(boolean)` on true, when `text` or `image` fail, it will fallback to the alternate method
* `setMinSizeBeforeFallback(int = 1)` number of characters to have at a minimum, before falling back.
* `setPageSegMode(int = 3)` image mode page segmentation mode
* `setEngineMode(int = 1)` image mode engine mode
* `setPageIteratorLevel(int = 0)` image mode page iteratior level
* `setScalingFactor(float)` Specifies the scaling factor to apply to images, in both axes, before OCR. It can scale up the image(factor > 1.0) or scale it down(factor < 1.0)
* `setSplitPages(boolean = true)` Whether to split pages into different rows and documents
* `setSplitRegions(boolean = true)` Whether to split by document regions. Works only in image mode. Enables split pages as well.
* `setIncludeConfidence(boolean = false)`
* `setAutomaticSkewCorrection(use: boolean, half_angle: double = 5.0, resolution: double = 1.0)`
* `setAutomaticSizeCorrection(use: boolean, desired_size: int = 34)`
* `setEstimateNoise(string)` image mode estimator noise level
* `useErosion(use: boolean, kernel_size: int = 2, kernel_shape: Int = 0)` image mode erosion