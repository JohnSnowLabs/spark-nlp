[![Build Status](https://travis-ci.org/JohnSnowLabs/spark-nlp.svg?branch=master)](https://travis-ci.org/JohnSnowLabs/spark-nlp)
# Spark-NLP
John Snow Labs Spark-NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

# Project's website
Take a look at our official spark-nlp page: http://nlp.johnsnowlabs.com/ for user documentation and examples

# Slack community channel
Questions? Feedback? Request access sending an email to nlp@johnsnowlabs.com

# Table of contents
* [Using Spark-NLP](#usage)
    * [Apache Spark Support](#apache-spark-support)
    * [Spark Packages](#spark-packages)
    * [Compiled JARs](#compiled-jars)
    * [Scala](#scala)
      * [Maven](#maven)
      * [SBT](#sbt)
    * [Python](#python)
    * [Apache Zeppelin](#apache-zeppelin)
    * [Jupyter Notebook](#jupyter-notebook-python)
    * [S3 Cluster](#s3-cluster)
 * [Offline Models](#offline-models)

## Usage

## Apache Spark Support
Spark-NLP *1.8.1* has been built on top of Apache Spark 2.4.0

Note that Spark is not retrocompatible with Spark 2.3.x, so models and environments might not work.

If you are still stuck on Spark 2.3.x feel free to use [this assembly jar](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-2.3.2-nlp-assembly-1.8.0.jar) instead. Support is limited.
For OCR module, [this](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-2.3.2-nlp-ocr-assembly-1.8.0.jar) is for spark 2.3.x 

| Spark NLP   |   Spark 2.0.0 / Spark 2.3.x         | Spark 2.4    |
|-------------|-------------------------------------|--------------|
| 1.8.x       |Partially (without pretrained models)|YES           |
| 1.7.3       |YES                                  |N/A           |
| 1.6.3       |YES                                  |N/A           |
| 1.5.0       |YES                                  |N/A           |

Find out more about `Spark-NLP` versions from our [release notes]().


## Spark Packages
#### Command line (requires internet connection)

This library has been uploaded to the spark-packages repository https://spark-packages.org/package/JohnSnowLabs/spark-nlp .

Benefit of spark-packages is that makes it available for both Scala-Java and Python

To use the most recent version just add the `--packages JohnSnowLabs:spark-nlp:1.8.1` to you spark command

```sh
spark-shell --packages JohnSnowLabs:spark-nlp:1.8.1
```

```sh
pyspark --packages JohnSnowLabs:spark-nlp:1.8.1
```

```sh
spark-submit --packages JohnSnowLabs:spark-nlp:1.8.1
```
## Compiled JARs
### Offline mode using jars

Either download pre-compiled packages [here](#pre-compiled-spark-nlp-and-spark-nlp-ocr) or build from source using `sbt assembly`

### Pre-compiled Spark-NLP and Spark-NLP-OCR
Spark-NLP FAT-JAR from here (Does NOT include Spark):
[Spark-NLP 1.8.1 FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-1.8.1.jar)

Spark-NLP GPU Enhanced Tensorflow FAT-JAR:
[Spark-NLP 1.8.1-gpu FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-1.8.1-gpu.jar)

Spark-NLP-OCR Module (Requires native Tesseract 4.x+ for image based OCR. Does not require Spark-NLP to work but highly suggested)
[Spark-NLP-OCR 1.8.1 FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-ocr-assembly-1.8.1.jar)

### Using the jar manually 

If for some reason you need to use the jar, you can download the jar from the project's website: http://nlp.johnsnowlabs.com/

From there you can use it in your project setting the `--classpath`

To add jars to spark programs use the `--jars` option

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
    <version>1.8.1</version>
</dependency>
```
and

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-ocr_2.11</artifactId>
    <version>1.8.1</version>
</dependency>
```

### SBT

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.8.1"

```
and

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-ocr" % "1.8.1"
```

Maven centeral: https://mvnrepository.com/artifact/com.johnsnowlabs.nlp

## Python
### Python without explicit Spark installation
If you installed pyspark through pip, you can install sparknlp through pip as well
```
pip install spark-nlp==1.8.1
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

## Apache Zeppelin
Use either one of the following options

* Add the following Maven Coordinates to the interpreter's library list
```
com.johnsnowlabs.nlp:spark-nlp_2.11:1.8.1
```
* Add path to pre-built jar from [here](#pre-compiled-spark-nlp-and-spark-nlp-ocr) in the interpreter's library list making sure the jar is available to driver path

### Python in Zeppelin
Apart from previous step, install python module through pip
```
pip install spark-nlp==1.8.1
```
Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and installed the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as shown earlier, since it includes both scala and python side installation.

## Jupyter Notebook (Python)

Easiest way to get this done is by making Jupyter Notebook run using pyspark as follows:

```
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages JohnSnowLabs:spark-nlp:1.8.1
```

Alternatively, you can mix in using `--jars` option for pyspark + `pip install spark-nlp`

If not using pyspark at all, you'll have to run the instructions pointed [here](#python-without-explicit-spark-installation)

## S3 Cluster
### With no hadoop configuration
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

## Offline Models

If you have troubles using pretrained() models in your environment, here a list to various models (only valid for latest versions).
If there is any older than current version of a model, it means they still work for current versions.
### Updated for 1.8.1
### Pipelines
* [Basic Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_basic_en_1.8.0_2.4_1545435998968.zip)
* [Advanced Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_advanced_en_1.8.0_2.4_1545436028146.zip)
* [Vivekn Sentiment Pipeline](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_vivekn_en_1.8.0_2.4_1545436008101.zip)

### Models
* [PerceptronModel (POS)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_fast_en_1.8.0_2.4_1545434653742.zip)
* [ViveknSentimentModel (Sentiment)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vivekn_fast_en_1.8.0_2.4_1545435741623.zip)
* [SymmetricDeleteModel (Spell Checker)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spell_sd_fast_en_1.8.0_2.4_1545435558025.zip)
* [ContextSpellCheckerModel (Spell Checker)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/context_spell_gen_en_1.8.0_2.4_1546979465177.zip)
* [NorvigSweetingModel (Spell Checker)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spell_fast_en_1.8.0_2.4_1545435732032.zip)
* [NerCRFModel (NER)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_fast_en_1.8.0_2.4_1545435254745.zip)
* [NerDLModel (NER)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_precise_en_1.8.0_2.4_1545439567330.zip)
* [LemmatizerModel (Lemmatizer)](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_fast_en_1.8.0_2.4_1545435317864.zip)

# FAQ
[Check our Articles and FAQ page here](https://nlp.johnsnowlabs.com/articles.html)

# Troubleshooting

### OCR
* Q: I am getting a Java Core Dump when running OCR transformation
  * A: Add `LC_ALL=C` environment variable
  
* Q: Getting `org.apache.pdfbox.filter.MissingImageReaderException: Cannot read JPEG2000 image: Java Advanced Imaging (JAI) Image I/O Tools are not installed` when running an OCR transformation
  * A: `--packages com.github.jai-imageio:jai-imageio-jpeg2000:1.3.0`. This library is non-free thus we can't include it as a Spark-NLP dependency by default


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
