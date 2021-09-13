---
layout: docs
header: true
title: Spark OCR 2.3.x (Licensed)
permalink: /docs/en/ocr_old
key: docs-ocr-old
modify_date: "2020-02-04"
sidebar:
    nav: spark-ocr
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
import com.johnsnowlabs.nlp._
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

## Utilizing Spark NLP OCR Module

Spark NLP OCR Module is not included within Spark NLP. It is not an
annotator and not an extension to Spark ML.

You can use OcrHelper to directly create spark dataframes from PDF.
This will hold entire documents in single rows, meant to be later
processed by a SentenceDetector. This way, you won't be breaking the
content in rows as if you were reading a standard document. Metadata
columns are added automatically and will include page numbers, file
name and other useful information per row.

***Python code***

```python
from pyspark.sql import SparkSession
from sparknlp.ocr import OcrHelper
from sparknlp import DocumentAssembler

data = OcrHelper().createDataset(spark = spark, input_path = "/your/example.pdf" )
documentAssembler = DocumentAssembler().setInputCol("text")
annotations = documentAssembler.transform(data)
annotations.columns
```

```bash
['text', 'pagenum', 'method', 'noiselevel', 'confidence', 'positions',
 'filename', 'document']
```

***Scala code***

```scala
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.DocumentAssembler

val myOcrHelper = new OcrHelper
val data = myOcrHelper.createDataset(spark, "/your/example.pdf")
val documentAssembler = new DocumentAssembler().setInputCol("text")
val annotations = documentAssembler.transform(data)
annotations.columns
```

```bash
Array[String] = Array(text, pagenum, method, noiselevel, confidence, positions, filename, document)
```

... where the text column of the annotations spark dataframe includes the text content of the PDF, pagenum the page number, etc...

### Creating an Array of Strings from PDF (For LightPipeline)

Another way, would be to simply create an array of strings. This is
useful for example if you are parsing a small amount of pdf files and
would like to use LightPipelines instead. See an example below.

***Scala code***

```scala
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.{DocumentAssembler,LightPipeline}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import org.apache.spark.ml.Pipeline

val myOcrHelper = new OcrHelper
val raw = myOcrHelper.createMap("/pdfs/")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val lightPipeline = new LightPipeline(new Pipeline().setStages(Array(documentAssembler, sentenceDetector)).fit(Seq.empty[String].toDF("text")))
val annotations = ligthPipeline.annotate(raw.values.toArray)
```

Now to get the whole first PDF content in your **/pdfs/** folder you can
use:

```scala
annotations(0)("document")(0)
```

and to get the third sentence found in that first pdf:

```scala
annotations(0)("sentence")(2)
```

To get from the fifth pdf the second sentence:

```scala
annotations(4)("sentence")(1)
```

Similarly, the whole content of the fifth pdf can be retrieved by:

```scala
annotations(4)("document")(0)
```
