---
layout: article
title: Licensed Installation
permalink: /docs/en/licensed_install
key: docs-licensed-install
modify_date: "2020-03-30"
---

### Install Licensed Spark NLP

You can also install the licensed package with extra functionalities and
pretrained models by using:

```bash
pip install spark-nlp-jsl==2.4.2 --extra-index-url #### --ignore-installed
```

The #### is a secret url only avaliable for users with license, if you
have not received it please contact us at info@johnsnowlabs.com.

At the moment there is no conda package for Licensed Spark NLP version.

### Setup AWS-CLI Credentials for licensed pretrained models

From Licensed version 2.4.2 in order to access private JohnSnowLabs
models repository you need first to setup your AWS credentials. This
access is done via Amazon aws command line interface (AWSCLI).

Instructions about how to install awscli are available at:

[https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)

Make sure you configure your credentials with aws configure following
the instructions at:

[https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)

Please substitute the ACCESS_KEY and SECRET_KEY with the credentials you
have recived. If you need your credentials contact us at
info@johnsnowlabs.com

### Start Licensed Spark NLP Session from python

The following will initialize the spark session in case you have run
the jupyter notebook directly. If you have started the notebook using
pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1
minute) as the jar from the server needs to be loaded.

The #### in .config("spark.jars", "####") is a secret code, if you have
not received it please contact us at info@johnsnowlabs.com.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Global DEMO - Spark NLP Enterprise 2.4.2") \
    .master("local[*]") \
    .config("spark.driver.memory","16") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4") \
    .config("spark.jars", "####") \
    .getOrCreate()
```
