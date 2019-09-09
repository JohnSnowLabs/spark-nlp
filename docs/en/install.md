---
layout: article
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2019-05-16"
---

## Spark-NLP Python

### Install OpenSource spark-nlp and pyspark pip packages

Be sure that you have the required python libraries (pyspark 2.4.3,
spark-nlp 2.2.1) by running 
```bash
pip list
```

Check that the versions are correct:
* pyspark 2.4.3
* spark-nlp 2.2.1

If some of them is missing you can run:
```bash
pip install --ignore-installed pyspark==2.4.3
pip install --ignore-installed spark-nlp==2.2.1
```
The --ignore-installed parameter is to overwrite your previous pip
package version if already installed.

### Install Licensed spark-nlp pip package

You can also install the licensed package with extra functionalities and
pretrained models. Check that spark-nlp-jsl 2.2.1 is installed by
running:
```bash 
pip install
```
If it is not then you need to install it by using:
```bash
pip install spark-nlp-jsl==2.2.1 --extra-index-url #### --ignore-installed
```
The #### is a secret url, if you have not received it please contact us 
at info@johnsnowlabs.com.

### Conda

If you are using Anaconda/Conda for managing Python packages, you can 
install `spark-nlp` as follow:

```bash
conda install -c johnsnowlabs spark-nlp
```

Anaconda [spark-nlp package](https://anaconda.org/JohnSnowLabs/spark-nlp)

Then you'll have to create a SparkSession manually, for example:

```bash
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[*]")\
    .config("spark.driver.memory","8G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.2.1")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()
```
If using local jars, you can use `spark.jars` instead for a comma
delimited jar files. For cluster setups, of course you'll have to put
the jars in a reachable location for all driver and executor nodes.

## Setup AWS-CLI Credentials for licensed pretrained models 

In order to access private JohnSnowLabs models repository you need first
to setup your AWS credentials. This access is done via Amazon aws
command line interface (AWSCLI).

Instructions about how to install awscli are available at:

[https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html]

Make sure you configure your credentials with aws configure following
the instructions at:

[https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html]

Please substitute the ACCESS_KEY and SECRET_KEY with the credentials you
have recived. If you need your credentials contact us at
info@johnsnowlabs.com


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

As an existing or experienced Python user, you may wish to install
Jupyter using Python’s package manager, pip, instead of Anaconda.

If you have Python 3 installed (which is recommended):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```
Congratulations, you have installed Jupyter Notebook! To run the
notebook, run the following command at the Terminal (Mac/Linux) or
Command Prompt (Windows):

```bash
jupyter notebook
```

## Spark-NLP Scala

Our package is deployed to maven central. In order to add this package
as a dependency in your application:

### Maven

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.2.1</version>
</dependency>
```

and

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-ocr_2.11</artifactId>
    <version>2.2.1</version>
</dependency>
```

### SBT

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.2.1"
```

and

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-ocr
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-ocr" % "2.2.1"
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

5- Now, all available **Spark Packages** are at your fingertips! Just search for **JohnSnowLabs:spark-nlp:version** where **version** stands for the library version such as: `1.8.4` or `2.2.1`
![Databricks](https://databricks.com/wp-content/uploads/2015/07/browser-1024x548.png)

6- Select **spark-nlp** package and we are good to go!

More info about how to use 3rd [Party Libraries in Databricks](https://databricks.com/blog/2015/07/28/using-3rd-party-libraries-in-databricks-apache-spark-packages-and-maven-libraries.html)
