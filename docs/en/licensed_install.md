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
<br/>
<br/>
Spark NLP for Healthcare provides healthcare-specific annotators, pipelines, models, and embeddings for:
- Clinical entity recognition
- Clinical Entity Linking
- Entity normalization
- Assertion Status Detection
- De-identification
- Relation Extraction
- Spell checking & correction


The library offers access to several clinical and biomedical transformers: JSL-BERT-Clinical, BioBERT, ClinicalBERT, GloVe-Med, GloVe-ICD-O. It also includes over 50 pre-trained healthcare models, that can recognize the following entities:
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
pip install spark-nlp-jsl==2.6.0 --extra-index-url {secret-url} --upgrade
```


The `{secret-url}` is a secret URL only available for users with valid/trial license. If you did not receive it yet, please contact us at <a href="mailto:info@johnsnowlabs.com">info@johnsnowlabs.com</a>.




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

The `{secret-url}` / `{secret-code}` tokens are secret  strings you should have received from your Customer Owner (CO). If you have
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
    .config("spark.jars", "{secret-url}") \
    .getOrCreate()
```