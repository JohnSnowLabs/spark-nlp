[![Build Status](https://travis-ci.org/JohnSnowLabs/spark-nlp.svg?branch=master)](https://travis-ci.org/JohnSnowLabs/spark-nlp)
# Spark-NLP
John Snow Labs Spark-NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

# Project's website
Take a look at our official spark-nlp page: http://nlp.johnsnowlabs.com/ for user documentation and examples

# Slack community channel
Questions? Feedback? Request access sending an email to nlp@johnsnowlabs.com

# Usage

## Command line
### with internet connection

This library has been uploaded to the spark-packages repository https://spark-packages.org/package/JohnSnowLabs/spark-nlp .

Benefit of spark-packages is that makes it available for both Scala-Java and Python

To use the most recent version just add the `--packages JohnSnowLabs:spark-nlp:1.7.3` to you spark command

```sh
spark-shell --packages JohnSnowLabs:spark-nlp:1.7.3
```

```sh
pyspark --packages JohnSnowLabs:spark-nlp:1.7.3
```

```sh
spark-submit --packages JohnSnowLabs:spark-nlp:1.7.3
```

### offline mode using jars

Either download pre-compiled packages [here](#pre-compiled-spark-nlp-and-spark-nlp-ocr) or build from source using `sbt assembly`

## Apache Zeppelin
Use either one of the following options

* Add the following Maven Coordinates to the interpreter's library list
```
com.johnsnowlabs.nlp:spark-nlp_2.11:1.7.3
```
* Add path to pre-built jar from [here](#pre-compiled-spark-nlp-and-spark-nlp-ocr) in the interpreter's library list making sure the jar is available to driver path

### Python in Zeppelin
Apart from previous step, install python module through pip
```
pip install spark-nlp==1.7.3
```
Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and installed the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as shown earlier, since it includes both scala and python side installation.

## Python without explicit Spark installation
If you installed pyspark through pip, you can install sparknlp through pip as well
```
pip install spark-nlp==1.7.3
```
Then you'll have to create a SparkSession manually, for example:
```
spark = SparkSession.builder \
    .appName("ner")\
    .master("local[4]")\
    .config("spark.driver.memory","4G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.driver.extraClassPath", "lib/sparknlp.jar")\
    .config("spark.executor.extraClassPath", "lib/sparknlp.jar")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()
```
For cluster setups, of course you'll have to put the jars in a reachable location for all driver and executor nodes

## Jupyter Notebook (Python)

Easiest way to get this done is by making Jupyter Notebook run using pyspark as follows:

```
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages JohnSnowLabs:spark-nlp:1.7.3
```

Alternatively, you can mix in using `--jars` option for pyspark + `pip install spark-nlp`

If not using pyspark at all, you'll have to run the instructions pointed [here](#python-without-explicit-spark-installation)

## S3 Cluster with no hadoop configuration
If your distributed storage is S3 and you don't have a standard hadoop configuration (i.e. fs.defaultFS)
You need to specify where in the cluster distributed storage you want to store Spark-NLP's tmp files.
First, decide where you want to put your *application.conf* file
```
import com.johnsnowlabs.uti.ConfigLoader
ConfigLoader.setConfigPath("/somewhere/to/put/application.conf")
```
And then we need to put in such application.conf the following content
```
sparknlp {
  settings {
    cluster_tmp_dir = "somewhere in s3n:// path to some folder"
  }
}
```

## Pre-compiled Spark-NLP and Spark-NLP-OCR
You may download fat-jar from here:
[Spark-NLP 1.7.3 FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-1.7.3.jar)
or non-fat from here
[Spark-NLP 1.7.3 PKG JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp_2.11-1.7.3.jar)
Spark-NLP-OCR Module (Requires native Tesseract 4.x+ for image based OCR. Does not require Spark-NLP to work but highly suggested)
[Spark-NLP-OCR 1.7.3 FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-ocr-assembly-1.7.3.jar)

## Maven central

Our package is deployed to maven central. In order to add this package as a dependency in your application:

#### Maven

```xml
<dependency>
  <groupId>com.johnsnowlabs.nlp</groupId>
  <artifactId>spark-nlp_2.11</artifactId>
  <version>1.7.3</version>
</dependency>
```

#### SBT
```sbtshell
libraryDependencies += "com.johnsnowlabs.nlp" % "spark-nlp_2.11" % "1.7.3"
```

If you are using `scala 2.11`

```sbtshell
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.7.3"
```

## Using the jar manually 

If for some reason you need to use the jar, you can download the jar from the project's website: http://nlp.johnsnowlabs.com/

From there you can use it in your project setting the `--classpath`

To add jars to spark programs use the `--jars` option

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in the `spark-packages` section.

## Downloading models for offline use

If you have troubles using pretrained() models in your environment, here a list to various models (only valid for latest versions).
If there is any older than current version of a model, it means they still work for current versions.
### Updated for 1.7.3
### Pipelines
* [Basic Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_basic_en_1.6.1_2_1533856444797.zip)
* [Advanced Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_advanced_en_1.7.0_2_1539460910585.zip)
* [Vivekn Sentiment Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_vivekn_en_1.6.2_2_1534781342094.zip)

### Models
* [PerceptronModel (POS)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_fast_en_1.6.1_2_1533853928168.zip)
* [ViveknSentimentModel (Sentiment)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vivekn_fast_en_1.6.2_2_1534781337758.zip)
* [SymmetricDeleteModel (Spell Checker)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spell_sd_fast_en_1.6.2_2_1534781178138.zip)
* [NorvigSweetingModel (Spell Checker)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spell_fast_en_1.6.2_2_1534781328404.zip)
* [AssertionDLModel (Assertion Status)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/as_fast_dl_en_1.6.1_2_1533855787457.zip)
* [NerCRFModel (NER)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_fast_en_1.7.0_2_1539896043754.zip)
* [NerDLModel (NER)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_precise_en_1.7.0_2_1539623388047.zip)
* [LemmatizerModel (Lemmatizer)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fast_en_1.6.1_2_1533854538211.zip)
* [AssertionDLModel (Assertion)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/as_fast_dl_en_1.7.0_2_1539653960749.zip)

# FAQ
[Check our Articles and FAQ page here](https://nlp.johnsnowlabs.com/articles.html)

# Special community aknowledgments
Thanks in general to the community who have been lately reporting important issues and pull request with bugfixes.
Community has been key in the last releases with feedback in various Spark based environments.

Here a few specific mentions for recurring feedback and slack participation
* @maziyarpanahi (https://github.com/maziyarpanahi) - For contributing with testing and valuable feedback
* @easimadi (https://github.com/easimadi) - For contributing with documentation and valuable feedback

# Contribute
We appreciate any sort of contributions:
* ideas
* feedback
* documentation
* bug reports
* nlp training and testing corpora
* development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

# Contact
nlp@johnsnowlabs.com

# John Snow Labs
http://johnsnowlabs.com/
