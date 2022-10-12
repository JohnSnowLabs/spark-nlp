---
layout: docs
header: true
title: Examples
key: docs-examples
permalink: /docs/en/examples
modify_date: "2021-11-21"
---

Showcasing notebooks and codes of how to use Spark NLP in Python and Scala.

## Python Setup

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.7 -y
$ conda activate sparknlp
$ pip install spark-nlp==4.2.1 pyspark==3.2.1
```

## Google Colab Notebook

Google Colab is perhaps the easiest way to get started with spark-nlp. It requires no installation or setup other than having a Google account.

Run the following code in Google Colab notebook and start using spark-nlp right away.

```sh
# This is only to setup PySpark and Spark NLP on Colab
!wget http://setup.johnsnowlabs.com/colab.sh -O - | bash
```

This script comes with the two options to define `pyspark` and `spark-nlp` versions via options:

```sh
# -p is for pyspark
# -s is for spark-nlp
# by default they are set to the latest
!bash colab.sh -p 3.2.1 -s 4.2.1
```

[Spark NLP quick start on Google Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs named entity recognitions and sentiment analysis by using Spark NLP pretrained pipelines.

## Kaggle Kernel

Run the following code in Kaggle Kernel and start using spark-nlp right away.

```sh
# Let's setup Kaggle for Spark NLP and PySpark
!wget http://setup.johnsnowlabs.com/kaggle.sh -O - | bash
```

## Notebooks

* [Tutorials and trainings](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials)
* [Jupyter Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter)
* [Databricks Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks)
