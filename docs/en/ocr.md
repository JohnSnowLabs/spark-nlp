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
