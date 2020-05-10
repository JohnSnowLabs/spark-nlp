---
layout: article
title: Licensed Installation
permalink: /docs/en/licensed_install
key: docs-licensed-install
modify_date: "2020-04-21"
---

### Install Licensed Spark NLP

You can also install the licensed package with extra functionalities and
pretrained models by using:

```bash
pip install spark-nlp-jsl==2.4.6 --extra-index-url {secret-url} --upgrade
```

The `{secret-url}` is a secret URL only available for users with valid license. If you
have not received it, please contact us at <a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.

At the moment there is no conda package for Licensed Spark NLP version.

### Setup AWS-CLI Credentials for licensed pretrained models

Starting from Licensed version 2.4.2, you need to first setup your AWS credentials 
to be able to access the private repository for John Snow Labs Pretrained Models. 
You can do this setup via Amazon AWS Command Line Interface (AWSCLI).

Instructions about how to install AWSCLI are available at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html">Installing the AWS CLI</a>

Make sure you configure your credentials with aws configure following
the instructions at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html">Configuring the AWS CLI</a>

Please substitute the `ACCESS_KEY` and `SECRET_KEY` with the credentials you
have received from your Customer Owner (CO). If you need your credentials contact us at 
<a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.

### Start Licensed Spark NLP Session from Python

The following will initialize the spark session in case you have run
the jupyter notebook directly. If you have started the notebook using
pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1
minute) as the jar from the server needs to be loaded.

The `{secret-url}` / `{secret-code}` tokens are secret 
strings you should have received from your Customer Owner (CO). If you have
not received them, please contact us at <a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.

You can either use our convenience function to start your Spark Session that will use standard configuration arguments:

```python
import sparknlp_jsl
spark = sparknlp_jsl.start("{secret.code}")
```

Or use the SparkSession module for more flexibility:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spark NLP Enterprise 2.4.6 Session") \
    .master("local[*]") \
    .config("spark.driver.memory","16") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5") \
    .config("spark.jars", "{secret-url}") \
    .getOrCreate()
```
