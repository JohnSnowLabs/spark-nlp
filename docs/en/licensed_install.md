---
layout: docs
header: true
title: Spark NLP for Healthcare
permalink: /docs/en/licensed_install
key: docs-licensed-install
modify_date: "2020-09-09"
---

<div class="h3-box" markdown="1">

### Getting started

Spark NLP for Healthcare is a commercial extension of Spark NLP for clinical and biomedical text mining. If you don't have a Spark NLP for Healthcare subscription yet, you can ask for a free trial by clicking on the button below.

{:.btn-block}
[Try Free](https://www.johnsnowlabs.com/spark-nlp-try-free/){:.button.button--primary.button--rounded.button--lg}


Spark NLP for Healthcare provides healthcare-specific annotators, pipelines, models, and embeddings for:
- Clinical entity recognition
- Clinical Entity Linking
- Entity normalization
- Assertion Status Detection
- De-identification
- Relation Extraction
- Spell checking & correction

note: If you are going to use any pretrained licensed NER model, you don't need to install licensed libray. As long as you have the AWS keys and license keys in your environment, you will be able to use licensed NER models with Spark NLP public library. For the other licensed pretrained models like AssertionDL, Deidentification, Entity Resolvers and Relation Extraction models, you will need to install Spark NLP Enterprise as well.

The library offers access to several clinical and biomedical transformers: JSL-BERT-Clinical, BioBERT, ClinicalBERT, GloVe-Med, GloVe-ICD-O. It also includes over 50 pre-trained healthcare models, that can recognize the following entities (any many more):
- Clinical - support Signs, Symptoms, Treatments, Procedures, Tests, Labs, Sections
- Drugs - support Name, Dosage, Strength, Route, Duration, Frequency
- Risk Factors- support Smoking, Obesity, Diabetes, Hypertension, Substance Abuse
- Anatomy - support Organ, Subdivision, Cell, Structure Organism, Tissue, Gene, Chemical
- Demographics - support Age, Gender, Height, Weight, Race, Ethnicity, Marital Status, Vital Signs
- Sensitive Data- support Patient Name, Address, Phone, Email, Dates, Providers, Identifiers


<br/>

### Install Spark NLP for Healthcare

You can install the Spark NLP for Healthcare package by using:

```bash
pip install -q spark-nlp-jsl==${version} --extra-index-url https://pypi.johnsnowlabs.com/${secret.code} --upgrade
```

`{version}` is the version part of the `{secret.code}` (`{secret.code}.split('-')[0]`) (i.e. `2.6.0`)

The `{secret.code}` is a secret code that is only available to users with valid/trial license. If you did not receive it yet, please contact us at <a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.


</div><div class="h3-box" markdown="1">

### Setup AWS-CLI Credentials for licensed pretrained models

Starting from Spark NLP for Healthcare version 2.4.2, you need to first setup your AWS credentials to be able to access the private repository for John Snow Labs Pretrained Models. 
You can do this setup via Amazon AWS Command Line Interface (AWSCLI).

Instructions about how to install AWSCLI are available at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html">Installing the AWS CLI</a>

Make sure you configure your credentials with aws configure following the instructions at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html">Configuring the AWS CLI</a>

Please substitute the `ACCESS_KEY` and `SECRET_KEY` with the credentials you have received from your Customer Owner (CO). If you need your credentials contact us at 
<a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.

</div>

### Start Spark NLP for Healthcare Session from Python

The following will initialize the spark session in case you have run the jupyter notebook directly. If you have started the notebook using
pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1 minute) as the jar from the server needs to be loaded.

The `{secret-code}` is a secret  string you should have received from your Customer Owner (CO). If you have
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
    .appName("Spark NLP Enterprise") \
    .master("local[*]") \
    .config("spark.driver.memory","16") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.0") \
    .config("spark.jars", "https://pypi.johnsnowlabs.com/${secret.code}/spark-nlp-jsl-${version}.jar") \
    .getOrCreate()
```

### Install Spark NLP for Healthcare on Databricks

1. Create a cluster if you don't have one already
2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab, in `Spark.Config` box:

    ```bash
    spark.kryoserializer.buffer.max 1000M
    spark.serializer org.apache.spark.serializer.KryoSerializer
    ```
      -  Please add the following to the `Advanced Options -> Spark` tab, in `Environment Variables` box:

    ```bash
    AWS_ACCESS_KEY_ID=xxx
    AWS_SECRET_ACCESS_KEY=yyy
    SPARK_NLP_LICENSE=zzz
    ```

      -   (OPTIONAL) If the environment variables used to setup the AWS Access/Secret keys are conflicting with the credential provider chain in Databricks, you may not be able to access to other s3 buckets. To access both JSL repos with JSL AWS keys as well as your own s3 bucket with your own AWS keys), you need to use the following script, copy that to dbfs folder, then go to the Databricks console (init scripts menu) to add the init script for your cluster as follows: 

    ```bash
    %scala
    val script = """
    #!/bin/bash

    echo "******** Inject Spark NLP AWS Profile Credentials ******** "

    mkdir ~/.aws/

    cat << EOF > ~/.aws/credentials
    [spark_nlp]
    aws_access_key_id=<YOUR_AWS_ACCESS_KEY>
    aws_secret_access_key=<YOUR_AWS_SECRET_KEY>
    EOF

    echo "******** End Inject Spark NLP AWS Profile Credentials  ******** "

    """
    ```

3. In `Libraries` tab inside your cluster you need to follow these steps:
 - Install New -> PyPI -> `spark-nlp` -> Install
 - Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.1` -> Install
 - Please add following jars:
        - Install New -> Python Whl -> upload `https://pypi.johnsnowlabs.com/${secret.code}/spark-nlp-jsl/spark_nlp_jsl-${version}-py3-none-any.whl`
        - Install New -> Jar -> upload `https://pypi.johnsnowlabs.com/${secret.code}/spark-nlp-jsl-${version}.jar`
4. Now you can attach your notebook to the cluster and use Spark NLP!
