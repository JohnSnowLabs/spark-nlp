---
layout: docs
header: true
title: Spark NLP - Examples
key: docs-examples
permalink: /docs/en/examples
modify_date: "2022-12-21"
---

<div class="h3-box" markdown="1">

Showcasing notebooks and codes of how to use Spark NLP in Python and Scala.

## Python Setup

```bash
$ java -version
# Java 8 or 11 for Apache Spark 3.x, Java 17, 21, or 25 for Apache Spark 4.x
$ conda create -n sparknlp python=3.9 -y
$ conda activate sparknlp

# Apache Spark 3.x (Scala 2.12)
$ pip install spark-nlp=={{ site.sparknlp_version }} pyspark==3.5.1

# Apache Spark 4.x (Scala 2.13)
$ pip install spark-nlp=={{ site.sparknlp_version }} pyspark==4.0.0
```

</div><div class="h3-box" markdown="1">

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
# Apache Spark 3.x
!bash colab.sh -p 3.5.1 -s {{ site.sparknlp_version }}
# Apache Spark 4.x
!bash colab.sh -p 4.0.0 -s {{ site.sparknlp_version }}
```

[Spark NLP quick start on Google Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs named entity recognitions and sentiment analysis by using Spark NLP pretrained pipelines.

</div><div class="h3-box" markdown="1"

## Kaggle Kernel

Run the following code in Kaggle Kernel and start using spark-nlp right away.

```sh
# Let's setup Kaggle for Spark NLP and PySpark
!wget http://setup.johnsnowlabs.com/kaggle.sh -O - | bash
```

</div><div class="h3-box" markdown="1">

## Notebooks

* [Tutorials and articles](https://medium.com/spark-nlp)
* [Jupyter Notebooks](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples)

</div>