---
layout: article
title: Quick Start
permalink: /docs/en/quickstart
key: docs-quickstart
modify_date: "2020-02-20"
---

## First: Join our Slack channel

Join our channel, to ask for help and share your feedback. Developers and users can help each other getting started here.

[Spark NLP Slack](https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLWVjNWUzOGNlODg1Y2FkNGEzNDQ1NDJjMjc3Y2FkOGFmN2Q3ODIyZGVhMzU0NGM3NzRjNDkyZjZlZTQ0YzY1N2I){:.button.button--info.button--rounded.button--md}

## Second: Spark NLP Workshop

If you prefer learning by example, check this repository:

[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop){:.button.button--primary.button--rounded.button--md}

It is full of fresh examples and even a docker container if you want to skip installation.

Below, you can follow into a more theoretical and thorough quick start guide.

## Requirements & Setup

Spark NLP is built on top of **Apache Spark 2.4.5**. This is the **only** supported release.

It is recommended to have basic knowledge of the framework and a working environment before using Spark NLP. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.4/index.html) to get started with Spark.

To start using the library, execute any of the following lines depending on your desired use case:

```bash
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

### Straight forward Python on jupyter notebook

As a first step we import the required python dependences including some sparknlp components.

Be sure that you have the required python libraries (pyspark 2.4.4, spark-nlp 2.4.5) by running pip list. Check that the versions are correct.

If some of them is missing you can run:

```bash
pip install --ignore-installed pyspark==2.4.
pip install --ignore-installed spark-nlp==2.4.5
```

Of course you will need to have jupyter installed in your system:

```bash
pip install jupyter
```

Now you should be ready to create a jupyter notebook running from terminal:

```bash
jupyter notebook
```

The easiest way to get started, is to run the following code:

```python
import sparknlp
sparknlp.start()
```

After a few seconds you should see something like this in your notebook:

```python
Version
    v2.4.5
Master
    local[*]
AppName
    Spark NLP
```

With those lines of code, you have successfully started a Spark Session and are ready to use Spark NLP

If you need more fine tuning, you will have to start SparkSession in your python program manually, for example

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .appName('Spark NLP') \
    .config("spark.driver.memory", "16g") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5") \
    .getOrCreate()
```

### Python Jupyter Notebook with PySpark

You can also run the Jupyter Notebook directly from Pyspark. In such
case you don't need to open a session, it will be automatically started 
by pyspark. Just remember to set the SPARK_HOME, PYSPARK_DRIVER_PYTHON and PYSPARK_DRIVER_PYTHON_OPTS environment variables.

```python
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook
```

To locate your SPARK_FOLDER you can for example run the following in a
linux system:

```bash
sudo find -wholename */jars/spark-core_*-2.4.4.jar
```

The parent folder where this ./jars/spark-core*-2.4.4.jar is your
SPARK_HOME folder.

In **Microsoft Windows** systems you can search for that file location in the explorer.

Once you have setup those environmental variables you can start a jupyter
notebook with a Spark (including sparknlp) session directly opened by
running in your terminal:

```bash
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

### Spark NLP from Scala

You can start a spark REPL with Scala by running in your terminal a
spark-shell including the com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5 package:

```bash
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

### Databricks cloud cluster & Apache Zeppelin

Add the following maven coordinates in the dependency configuration page:

```bash
com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

For Python in **Apache Zeppelin** you may need to setup _**SPARK_SUBMIT_OPTIONS**_ utilizing --packages instruction shown above like this

```bash
export SPARK_SUBMIT_OPTIONS="--packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5"
```

### S3 based standalone cluster (No Hadoop)

If your distributed storage is S3 and you don't have a standard hadoop configuration (i.e. fs.defaultFS) You need to specify where in the cluster distributed storage you want to store Spark NLP's tmp files. First, decide where you want to put your **application.conf** file

```bash
import com.johnsnowlabs.util.ConfigLoader
ConfigLoader.setConfigPath("/somewhere/to/put/application.conf")
```

And then we need to put in such application.conf the following content

```json
sparknlp {
    settings {
        cluster_tmp_dir = "somewhere in s3n:// path to some folder"
    }
}
```

For further alternatives and documentation check out our README page in [GitHub](https://github.com/JohnSnowLabs/spark-nlp).

## Where to go next

### Documentation and reference

If you need more detailed information about how to install Spark NLP you can check the [Installation page](install)

Detailed information about Spark NLP concepts, annotators and more may
be found [HERE](annotators)

### More examples in Scala and Python

We are working on examples to show you how the library may be used in
different scenarios, take a look at our examples repository, which also
includes a Docker image:

[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop){:.button.button--primary.button--rounded.button--md}
