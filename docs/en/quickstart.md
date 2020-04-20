---
layout: article
title: Quick Start
permalink: /docs/en/quickstart
key: docs-quickstart
modify_date: "2020-04-06"
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

Spark NLP is built on top of **Apache Spark 2.4.4**. In order to use Spark NLP you need the following requirements:

* Java 8
* Apache Spark 2.4.x

It is recommended to have basic knowledge of the framework and a working environment before using Spark NLP. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.4/index.html) to get started with Spark.

To start using the library, execute any of the following lines depending on your desired use case:

```bash
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

### Straight forward Python on jupyter notebook

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.6 -y
$ conda activate sparknlp
$ pip install spark-nlp==2.4.5 pyspark==2.4.4
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
    .appName("Spark NLP")\
    .master("local[4]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()
```

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
