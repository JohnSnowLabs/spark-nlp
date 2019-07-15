# Spark NLP

[![Build Status](https://travis-ci.org/JohnSnowLabs/spark-nlp.svg?branch=master)](https://travis-ci.org/JohnSnowLabs/spark-nlp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.johnsnowlabs.nlp/spark-nlp_2.11/badge.svg)](https://search.maven.org/artifact/com.johnsnowlabs.nlp/spark-nlp_2.11) [![PyPI version](https://badge.fury.io/py/spark-nlp.svg)](https://badge.fury.io/py/spark-nlp) [![Anaconda-Cloud](https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE)

John Snow Labs Spark NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

## Project's website

Take a look at our official Spark NLP page: [http://nlp.johnsnowlabs.com/](http://nlp.johnsnowlabs.com/) for user documentation and examples

## Slack community channel

[Join Slack](https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLTM4ZDliMjU5OWZmMDE1ZGVkMjg0MWFjMjU3NjY4YThlMTJkNmNjNjM3NTMwYzlhMWY4MGMzODI2NDBkOWU4ZDE)

## Table of contents

* [Using Spark NLP](#usage)
  * [Apache Spark Support](#apache-spark-support)
  * [Spark Packages](#spark-packages)
  * [Compiled JARs](#compiled-jars)
  * [Scala](#scala)
    * [Maven](#maven)
    * [SBT](#sbt)
  * [Python](#python)
    * [Pip/Conda](#pipconda)
  * [Apache Zeppelin](#apache-zeppelin)
  * [Jupyter Notebook](#jupyter-notebook-python)
  * [Google Colab Notebook](#google-colab-notebook)
  * [S3 Cluster](#s3-cluster)
  * [OCR Module](#ocr-module)
* [Pipelines & Models](#pipelines-and-models)
  * [Pipelines](#pipelines)
  * [Models](#models)
* [Examples](#examples)  
* [FAQ](#faq)
* [Troubleshooting](#troubleshooting)
* [Aknowledgments](#aknowledgments)
* [Contributing](#contributing)

## Usage

## Apache Spark Support

Spark NLP *2.1.0* has been built on top of Apache Spark 2.4.3

Note that Spark is not retrocompatible with Spark 2.3.x, so models and environments might not work.

If you are still stuck on Spark 2.3.x feel free to use [this assembly jar](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-2.3.2-nlp-assembly-1.8.0.jar) instead. Support is limited.
For OCR module, [this](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-2.3.2-nlp-ocr-assembly-1.8.0.jar) is for spark `2.3.x`.

| Spark NLP   |   Spark 2.3.x         | Spark 2.4    |
|-------------|-------------------------------------|--------------|
| 2.x.x       |YES                                   |YES           |
| 1.8.x       |Partially                            |YES           |
| 1.7.3       |YES                                  |N/A           |
| 1.6.3       |YES                                  |N/A           |
| 1.5.0       |YES                                  |N/A           |

Find out more about `Spark NLP` versions from our [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases).

## Spark Packages

### Command line (requires internet connection)

This library has been uploaded to the [spark-packages repository](https://spark-packages.org/package/JohnSnowLabs/spark-nlp).

Benefit of spark-packages is that makes it available for both Scala-Java and Python

To use the most recent version just add the `--packages JohnSnowLabs:spark-nlp:2.1.0` to you spark command

```sh
spark-shell --packages JohnSnowLabs:spark-nlp:2.1.0
```

```sh
pyspark --packages JohnSnowLabs:spark-nlp:2.1.0
```

```sh
spark-submit --packages JohnSnowLabs:spark-nlp:2.1.0
```

This can also be used to create a SparkSession manually by using the `spark.jars.packages` option in both Python and Scala

## Compiled JARs

### Build from source

#### Spark NLP

* FAT-JAR for CPU

```bash
sbt assembly
```

* FAT-JAR for GPU

```bash
sbt -Dis_gpu=true assembly
```

* Packaging the project

```bash
sbt package
```

#### Spark NLP OCR

Requires native Tesseract 4.x+ for image based OCR. Does not require Spark NLP to work but highly suggested

* FAT-JAR

```bash
sbt ocr/assembly
```

* Packaging the project

```bash
sbt ocr/package
```

### Using the jar manually

If for some reason you need to use the JAR, you can either download the Fat JARs provided here or download it from [Maven Central](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp).

To add JARs to spark programs use the `--jars` option:

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in the `spark-packages` section.

## Scala

Our package is deployed to maven central. In order to add this package as a dependency in your application:

### Maven

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.1.0</version>
</dependency>
```

and

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-ocr_2.11</artifactId>
    <version>2.1.0</version>
</dependency>
```

### SBT

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.1.0"
```

and

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-ocr" % "2.1.0"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

## Python

### Python without explicit Pyspark installation

### Pip/Conda

If you installed pyspark through pip/conda, you can install `spark-nlp` through the same channel.

Pip:
```bash
pip install spark-nlp==2.1.0
```
Conda:
```bash
conda install -c johnsnowlabs spark-nlp
```

PyPI [spark-nlp package](https://pypi.org/project/spark-nlp/) / Anaconda [spark-nlp package](https://anaconda.org/JohnSnowLabs/spark-nlp)

Then you'll have to create a SparkSession manually, for example:

```bash
spark = SparkSession.builder \
    .appName("ner")\
    .master("local[4]")\
    .config("spark.driver.memory","4G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.1.0")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()
```

If using local jars, you can use `spark.jars` instead for a comma delimited jar files. For cluster setups, of course you'll have to put the jars in a reachable location for all driver and executor nodes

## Apache Zeppelin

Use either one of the following options

* Add the following Maven Coordinates to the interpreter's library list

```bash
com.johnsnowlabs.nlp:spark-nlp_2.11:2.1.0
```

* Add path to pre-built jar from [here](#pre-compiled-spark-nlp-and-spark-nlp-ocr) in the interpreter's library list making sure the jar is available to driver path

### Python in Zeppelin

Apart from previous step, install python module through pip

```bash
pip install spark-nlp==2.1.0
```

Or you can install `spark-nlp` from inside Zeppelin by using Conda:

```bash
python.conda install -c johnsnowlabs spark-nlp
```

Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and installed the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as shown earlier, since it includes both scala and python side installation.

## Jupyter Notebook (Python)

Easiest way to get this done is by making Jupyter Notebook run using pyspark as follows:

```bash
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages JohnSnowLabs:spark-nlp:2.1.0
```

Alternatively, you can mix in using `--jars` option for pyspark + `pip install spark-nlp`

If not using pyspark at all, you'll have to run the instructions pointed [here](#python-without-explicit-Pyspark-installation)

## Google Colab Notebook
Google Colab is perhaps the easiest way to get started with spark-nlp. It requires no installation or set up other than having a Google account.

Run the following code in Google Colab notebook and start using spark-nlp right away. 

```bash 
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
!tar xf spark-2.4.3-bin-hadoop2.7.tgz
!pip install -q findspark
!pip install spark-nlp

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.3-bin-hadoop2.7"

import findspark
findspark.init()

import sparknlp
spark = sparknlp.start()
```

[Here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs sentiment analysis and NER using pretrained spark-nlp models.

## S3 Cluster

### With no hadoop configuration

If your distributed storage is S3 and you don't have a standard hadoop configuration (i.e. fs.defaultFS)
You need to specify where in the cluster distributed storage you want to store Spark NLP's tmp files.
First, decide where you want to put your *application.conf* file

```scala
import com.johnsnowlabs.util.ConfigLoader
ConfigLoader.setConfigPath("/somewhere/to/put/application.conf")
```

And then we need to put in such application.conf the following content

```bash
sparknlp {
  settings {
    cluster_tmp_dir = "somewhere in s3n:// path to some folder"
  }
}
```

## OCR Module

To include the OCR submodule in Spark NLP, you will need to add the following to your start up commands:

```
--repositories http://repo.spring.io/plugins-release
--packages JohnSnowLabs:spark-nlp:2.1.0,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.1.0,javax.media.jai:com.springsource.javax.media.jai.core:1.1.3
```

This way you will download the extra dependencies needed by our OCR submodule. The Python SparkSession equivalent is

```
spark = SparkSession.builder \ 
    .master('local[*]') \ 
    .appName('Spark NLP with OCR') \ 
    .config("spark.driver.memory", "6g") \ 
    .config("spark.executor.memory", "6g") \ 
    .config("spark.jars.repositories", "http://repo.spring.io/plugins-release") \ 
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.1.0,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.1.0,javax.media.jai:com.springsource.javax.media.jai.core:1.1.3") \ 
    .getOrCreate() 
```

## Pipelines and Models

### Pipelines

Please check our documentation for pre-trained [pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines)

### Models

Please check our documentation for pre-trained [models](https://nlp.johnsnowlabs.com/docs/en/models)

## Examples

Need more examples? Check out our dedicated repository to showcase Spark NLP use cases!

[spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop)

## FAQ

[Check our Articles and FAQ page here](https://nlp.johnsnowlabs.com/articles.html)

## Troubleshooting

### OCR

* Q: I am getting a Java Core Dump when running OCR transformation
  * A: Add `LC_ALL=C` environment variable
  
* Q: Getting `org.apache.pdfbox.filter.MissingImageReaderException: Cannot read JPEG2000 image: Java Advanced Imaging (JAI) Image I/O Tools are not installed` when running an OCR transformation
  * A: `--packages com.github.jai-imageio:jai-imageio-jpeg2000:1.3.0`. This library is non-free thus we can't include it as a Spark NLP dependency by default

## Acknowledgments

### Special community aknowledgments

Thanks in general to the community who have been lately reporting important issues and pull request with bugfixes.
Community has been key in the last releases with feedback in various Spark based environments.

Here a few specific mentions for recurring feedback and slack participation

* [@maziyarpanahi](https://github.com/maziyarpanahi) - For contributing with testing and valuable feedback
* [@easimadi](https://github.com/easimadi) - For contributing with documentation and valuable feedback

## Contributing

We appreciate any sort of contributions:

* ideas
* feedback
* documentation
* bug reports
* nlp training and testing corpora
* development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

## Contact

nlp@johnsnowlabs.com

## John Snow Labs

[http://johnsnowlabs.com](http://johnsnowlabs.com)
