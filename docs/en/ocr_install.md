---
layout: docs
header: true
title: Installation
permalink: /docs/en/ocr_install
key: docs-ocr-install
modify_date: "2020-04-08"
---

<div class="h3-box" markdown="1">


Spark OCR is built on top of **Apache Spark**. 
Currently, it supports 2.4.* and 2.3.* versions of Spark.

It is recommended to have basic knowledge of the framework and a working environment before using Spark OCR. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.4/index.html) to get started with Spark.


Spark OCR required:
 - Scala 2.11
 - Python 3.+ (in case using PySpark)
 - Spark 2.4.* or 2.3.*

Before you start, make sure that you have: 
- Spark OCR jar file (or secret for download it)
- Spark OCR python wheel file 
- License key

If you don't have a valid subscription yet and you want to test out the Spark OCR library press the button below:

{:.btn-block}
[Try Free](https://www.johnsnowlabs.com/spark-nlp-try-free/){:.button.button--primary.button--rounded.button--lg}



</div><div class="h3-box" markdown="1">

## Spark OCR from Scala

You can start a spark REPL with Scala by running in your terminal a spark-shell including the com.johnsnowlabs.nlp:spark-ocr_2.11:1.0.0 package:

```bash
spark-shell --jars ####
```

The #### is a secret url only avaliable for license users. If you have purchansed a license but did not receive it please contact us at info@johnsnowlabs.com.

</div>

### Start Spark OCR Session

The following code will initialize the spark session in case you have run the jupyter notebook directly. If you have started the notebook using pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1 minute) as the jar from the server needs to be loaded.

The #### in .config("spark.jars", "####") is a secret code, if you have not received it please contact us at info@johnsnowlabs.com.


```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession
    .builder()
    .appName("Spark OCR")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.driver.maxResultSize", "2G")
    .config("spark.jars", "####")
    .getOrCreate()
```

<div class="h3-box" markdown="1">

## Spark OCR from Python

### Install Python package

Install python package using pip:

```bash
pip install spark-ocr==1.8.0.spark24 --extra-index-url #### --ignore-installed
```

The #### is a secret url only avaliable for license users. If you have purchansed a license but did not receive it please contact us at info@johnsnowlabs.com.

</div><div class="h3-box" markdown="1">

### Start Spark OCR Session

#### Manually

```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Spark OCR") \
    .master("local[*]") \
    .config("spark.driver.memory", "4G") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars", "https://pypi.johnsnowlabs.com/####") \
    .getOrCreate()
```

</div><div class="h3-box" markdown="1">

#### Using Start function

Another way to initialize SparkSession with Spark OCR to use `start` function in Python.


Start function has following params:

{:.table-model-big}
| Param name | Type | Default | Description |
| --- | --- | --- | --- |
| secret | string | None | Secret for download Spark OCR jar file |
| jar_path | string | None | Path to jar file in case you need to run spark session offline |
| extra_conf | SparkConf | None | Extra spark configuration |
| master_url | string | local[*] | Spark master url |
| nlp_version | string | None | Spark NLP version for add it Jar to session |
| nlp_internal | boolean/string | None | Run Spark session with Spark NLP Internal if set to 'True' or specify version |
| nlp_secret| string | None| Secret for get Spark NLP Internal jar |
| keys_file | string | `keys.json`| Name of the json file with license, secret and aws keys |

For start Spark session with Spark NLP please specify version of it in `nlp_version` param.

Example:

```python
from sparkocr import start
   
spark = start(secret=secret, nlp_version="2.4.4")
```

</div>

## Databricks

The installation process to Databricks includes following steps: 

- Installing Spark OCR library to Databricks and attaching it to the cluster 
- Same step for Spark OCR python wheel file 
- Adding license key
- Adding cluster init script for install dependencies

Please look [databricks python helpers](ocr_structures#databricks-python-helpers) for simplify install init script.

Example notebooks:

 - [Spark OCR Databricks python notebooks](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/databricks/python) 
 - [Spark OCR Databricks Scala notebooks](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/databricks/scala)
 