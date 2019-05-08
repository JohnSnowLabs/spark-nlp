---
layout: article
title: Installing Spark NLP
permalink: /docs/en/install
key: docs-install
show_edit_on_github: true
aside:
  toc: true
sidebar:
  nav: docs-en
# mode: immersive
header:
  theme: dark
  background: '#123'
---

## Spark-NLP Python

There are two directories: `training` and `annotation`. Inside `traning` you will find all the examples which help you to train Spark-NLP models and pipelines. On the other hand, the examples inside `annotation` demonstrate how to use Spark-NLP annotators, pre-trained models, and pre-trained pipelines.

### Pip

If you installed pyspark through pip, you can install `spark-nlp` through pip as well.

```bash
pip install spark-nlp==2.0.1
```

PyPI [spark-nlp package](https://pypi.org/project/spark-nlp/)

### Conda

If you are using Anaconda/Conda for managing Python packages, you can install `spark-nlp` as follow:

```bash
conda install -c johnsnowlabs spark-nlp
```

Anaconda [spark-nlp package](https://anaconda.org/JohnSnowLabs/spark-nlp)

Then you'll have to create a SparkSession manually, for example:

```bash
spark = SparkSession.builder \
    .appName("ner")\
    .master("local[*]")\
    .config("spark.driver.memory","6G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.0.1")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()
```

If using local jars, you can use `spark.jars` instead for a comma delimited jar files. For cluster setups, of course you'll have to put the jars in a reachable location for all driver and executor nodes

## Setup Jupyter Notebook

### Prerequisite: Python

While Jupyter runs code in many programming languages, Python is a requirement
(Python 3.3 or greater, or Python 2.7) for installing the Jupyter Notebook itself.

### Installing Jupyter using Anaconda

We **strongly recommend** installing Python and Jupyter using the [Anaconda Distribution](https://www.anaconda.com/downloads),
which includes Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

First, download [Anaconda](https://www.anaconda.com/downloads). We recommend downloading Anaconda’s latest Python 3 version.

Second, install the version of Anaconda which you downloaded, following the instructions on the download page.

Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

```bash
jupyter notebook
```

### Installing Jupyter with pip

As an existing or experienced Python user, you may wish to install Jupyter using Python’s package manager, pip, instead of Anaconda.

If you have Python 3 installed (which is recommended):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```


Congratulations, you have installed Jupyter Notebook! To run the notebook, run
the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

```bash
jupyter notebook
```

Original reference: [https://jupyter.org/install](https://jupyter.org/install)

## Spark-NLP Scala

Our package is deployed to maven central. In order to add this package as a dependency in your application:

### Maven

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.0.1</version>
</dependency>
```

and

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-ocr_2.11</artifactId>
    <version>2.0.1</version>
</dependency>
```

### SBT

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.0.1"
```

and

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-ocr" % "2.0.1"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

## Spark-NLP Databricks

### Databricks Notebooks

You can view all the Databricks notebooks from this address:

[https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html)

Note: You can import these notebooks by using their URLs.

### How to use Spark-NLP library in Databricks

1- Right-click the Workspace folder where you want to store the library.

2- Select Create > Library.

3- Select where you would like to create the library in the Workspace, and open the Create Library dialog:

![Databricks](https://databricks.com/wp-content/uploads/2015/07/create-lib.png)

4- From the Source drop-down menu, select **Maven Coordinate:**
![Databricks](https://databricks.com/wp-content/uploads/2015/07/select-maven-1024x711.png)

5- Now, all available **Spark Packages** are at your fingertips! Just search for **JohnSnowLabs:spark-nlp:version** where **version** stands for the library version such as: `1.8.4` or `2.0.1`
![Databricks](https://databricks.com/wp-content/uploads/2015/07/browser-1024x548.png)

6- Select **spark-nlp** package and we are good to go!

More info about how to use 3rd [Party Libraries in Databricks](https://databricks.com/blog/2015/07/28/using-3rd-party-libraries-in-databricks-apache-spark-packages-and-maven-libraries.html)
