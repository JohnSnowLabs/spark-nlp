[![Build Status](https://travis-ci.org/JohnSnowLabs/spark-nlp.svg?branch=master)](https://travis-ci.org/JohnSnowLabs/spark-nlp)
# Spark-NLP
John Snow Labs Spark-NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

# Project's website
Take a look at our official spark-nlp page: http://nlp.johnsnowlabs.com/ for user documentation and examples

# Slack community channel
Questions? Feedback? Request access sending an email to nlp@johnsnowlabs.com

# Usage

## spark-packages

This library has been uploaded to the spark-packages repository https://spark-packages.org/package/JohnSnowLabs/spark-nlp .

To use the most recent version just add the `--packages JohnSnowLabs:spark-nlp:1.5.4` to you spark command

```sh
spark-shell --packages JohnSnowLabs:spark-nlp:1.5.4
```

```sh
pyspark --packages JohnSnowLabs:spark-nlp:1.5.4
```

```sh
spark-submit --packages JohnSnowLabs:spark-nlp:1.5.4
```

## Jupyter Notebook

```
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages JohnSnowLabs:spark-nlp:1.5.4
```

## Apache Zeppelin
This way will work for both Scala and Python
```
export SPARK_SUBMIT_OPTIONS="--packages JohnSnowLabs:spark-nlp:1.5.4"
```
Alternatively, add the following Maven Coordinates to the interpreter's library list
```
com.johnsnowlabs.nlp:spark-nlp_2.11:1.5.4
```

## Python without explicit Spark installation
If you installed pyspark through pip, you can now install sparknlp through pip
```
pip install --index-url https://test.pypi.org/simple/ spark-nlp==1.5.4
```
Then you'll have to create a SparkSession manually, for example:
```
spark = SparkSession.builder \
    .appName("ner")\
    .master("local[4]")\
    .config("spark.driver.memory","4G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.driver.extraClassPath", "lib/sparknlp.jar")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()
```

## Pre-compiled Spark-NLP
You may download fat-jar from here:
[Spark-NLP 1.5.4 FAT-JAR](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/spark-nlp-assembly-1.5.4.jar)
or non-fat from here
[Spark-NLP 1.5.4 PKG JAR](http://repo1.maven.org/maven2/com/johnsnowlabs/nlp/spark-nlp_2.11/1.5.4/spark-nlp_2.11-1.5.4.jar)

## Maven central

Our package is deployed to maven central. In order to add this package as a dependency in your application:

#### Maven

```xml
<dependency>
  <groupId>com.johnsnowlabs.nlp</groupId>
  <artifactId>spark-nlp_2.11</artifactId>
  <version>1.5.4</version>
</dependency>
```

#### SBT
```sbtshell
libraryDependencies += "com.johnsnowlabs.nlp" % "spark-nlp_2.11" % "1.5.4"
```

If you are using `scala 2.11`

```sbtshell
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.5.4"
```

## Using the jar manually 

If for some reason you need to use the jar, you can download the jar from the project's website: http://nlp.johnsnowlabs.com/

From there you can use it in your project setting the `--classpath`

To add jars to spark programs use the `--jars` option

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in the `spark-packages` section.

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
