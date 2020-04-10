---
layout: article
title: Installation
permalink: /docs/en/ocr_install
key: docs-ocr-install
modify_date: "2020-04-08"
---

# Requirements

Spark OCR is built on top of **Apache Spark 2.4.4**. 
We have custom build for 2.3.*.

It is recommended to have basic knowledge of the framework and a working environment before using Spark OCR. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.4/index.html) to get started with Spark.

Spark OCR required:
 - Scala 2.11
 - Python 3.+ (in case using PySpark)
 - Spark 2.4.4
 - Tesseract 4.1.+.

Before you start, make sure that you have: 
- Spark OCR jar file (or secret for download it)
- Spark OCR python wheel file 
- License key

If not please contact info@johnsnowlabs.com to get the library and free trial license. 

# Installing Tesseract

## Debian, Ubuntu

As mentioned above, if you are dealing with scanned images instead of text-selectable PDF files you need to install `tesseract 4.x+` on all the nodes in your cluster. Here is how you can install it on Ubuntu/Debian:

```bash
apt-get install tesseract-ocr
```

For check installation and version run:

```bash
tesseract -v
```

Output:
```bash
tesseract 4.1.1
 leptonica-1.79.0
  libgif 5.2.1 : libjpeg 9d : libpng 1.6.37 : libtiff 4.1.0 : zlib 1.2.11 : libwebp 1.1.0 : libopenjp2 2.3.1
 Found AVX2
 Found AVX
 Found FMA
 Found SSE
```

Some of distributive contain old version of Tesseract. You can 
install more fresh version using PPA:

```bash
sudo add-apt-repository ppa:alex-p/tesseract-ocr
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Or build it manually from source:

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
```

## Fedora, Red Hat, Suse, CentOS, Amazon Linux

For have fresh version of Tesseract need to build it:

```bash
sudo yum install -y clang++ gcc gcc-c++ libstdc++
sudo yum install -y autoconf automake libtool make autoconf-archive
sudo yum install -y pkg-config
sudo yum install -y libpng-devel
sudo yum install -y libjpeg8-dev
sudo yum install -y libtiff-devel
sudo yum install -y zlib1g-dev
sudo yum install -y git

# Download, build  and install leptonica
wget https://src.fedoraproject.org/lookaside/extras/leptonica/leptonica-1.74.2.tar.gz/sha512/dff6db0f5fd536b68482a3c468206cd7f9436a9de47ccf080e1329cc0fa8f250df6f9a07288d72048cb123339b79cbbfe58ff6e272736fd273d393518348b22e/leptonica-1.74.2.tar.gz

tar xvf leptonica-1.74.2.tar.gz
cd leptonica-1.74.2
./autobuild
./configure
make
sudo make install

sudo ln -s /usr/local/lib/liblept.so /usr/lib64/liblept.so
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# Download, build  and install tesseract
git clone --single-branch --branch 4.1.1 https://github.com/tesseract-ocr/tesseract.git
cd tesseract
git checkout 4.1.1
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

## Mac OS

On Mac OS easy to install Tesseract using [homebrew](https://brew.sh/):

```bash
brew install tesseract
```

# Spark OCR from Scala

You can start a spark REPL with Scala by running in your terminal a spark-shell including the com.johnsnowlabs.nlp:spark-ocr_2.11:1.0.0 package:

```bash
spark-shell --jars ####
```

The #### is a secret url only avaliable for license users. If you have purchansed a license but did not receive it please contact us at info@johnsnowlabs.com.

## Start Spark OCR Session

The following code will initialize the spark session in case you have run the jupyter notebook directly. If you have started the notebook using pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1 minute) as the jar from the server needs to be loaded.

The #### in .config("spark.jars", "####") is a secret code, if you have not received it please contact us at info@johnsnowlabs.com.


```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession
    .builder()
    .appName("Spark OCR")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.driver.maxResultSize", "2G")
    .config("spark.jars", "####")
    .getOrCreate()
```

# Spark OCR from Python

## Install Python package

Install python package using pip:

```bash
pip install spark-ocr==1.1.0 --extra-index-url #### --ignore-installed
```

The #### is a secret url only avaliable for license users. If you have purchansed a license but did not receive it please contact us at info@johnsnowlabs.com.

## Start Spark OCR Session

### Manually

```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Spark OCR") \
    .master("local[*]") \
    .config("spark.driver.memory", "4G") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars", "https://pypi.johnsnowlabs.com/####") \
    .getOrCreate()
```

### Using Start function

Another way to initialize SparkSession with Spark OCR to use `start` function in Python.


Start function has following params:

| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| secret | string | None | Secret for download Spark OCR jar file |
| jar_path | string | None | Path to jar file in case you need to run spark session offline |
| extra_conf | SparkConf | None | Extra spark configuration |
| nlp_version | string | None | Spark NLP version for add it Jar to session |


For start Spark session with Spark NLP please specify version of it in `nlp_version` param.

Example:

```python
from sparkocr import start
   
spark = start(secret=secret, nlp_version="2.4.4")
```


# Databricks

The installation process to Databricks includes following steps: 

- Installing Spark OCR library to Databricks and attaching it to the cluster 
- Same step for Spark OCR python wheel file 
- Adding license key
- Adding cluster init script for install dependencies

Please look [databricks python helpers](ocr_structures#databricks-python-helpers) for simplify install init script.

Example notebooks:

 - [Spark OCR Databricks python notebooks](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/databricks/python) 
 - [Spark OCR Databricks Scala notebooks](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/databricks/scala)
 