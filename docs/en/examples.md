---
layout: article
title: Examples
key: docs-examples
permalink: /docs/en/examples
modify_date: "2020-09-01"
---

Showcasing notebooks and codes of how to use Spark NLP in Python and Scala.

## Python Setup

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.6 -y
$ conda activate sparknlp
$ pip install spark-nlp pyspark==2.4.4
```

## Colab setup

```python
import os

# Install java
! apt-get update -qq
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! java -version

# Install pyspark
! pip install -q pyspark==2.4.6
! pip install -q spark-nlp
```

## Notebooks

* [Tutorials and trainings](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials)
* [Jupyter Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter)
* [Databricks Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks)
