---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Installation
permalink: /docs/en/licensed_install
key: docs-licensed-install
modify_date: "2021-03-09"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## Install NLP libraries on Ubuntu 

For installing John Snow Labs NLP libraries on an Ubuntu machine/VM please run the following command:

```bash
wget https://setup.johnsnowlabs.com/nlp/install.sh -O - | sudo bash -s -- -a PATH_TO_LICENSE_JSON_FILE -i -r
```

This script will install `Spark NLP`, `Spark NLP for Healthcare`, `Spark OCR`, `NLU` and `Spark NLP Display` on the specified virtual environment. It will also create a special folder, `./JohnSnowLabs`,  dedicated to all resources necessary for using the libraries.  Under `./JohnSnowLabs/example_notebooks` you will find some ready to use example notebooks that you can use to test the libraries on your data. 

For a complete step by step guide on how to install NLP Libraries check the video below:
<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='E-zAkeym06g' -%}<div class="video-descr">Install John Snow Labs NLP Libraries on Ubuntu</div></div></div>


The install script offers several options:
 - `-h`    show brief help
 - `-i`    install mode: create a virtual environment and install the library
 - `-r`    run mode: start jupyter after installation of the library 
 - `-v`    path to virtual environment (default: ./sparknlp_env)
 - `-j`    path to license json file for Spark NLP for Healthcare
 - `-o`    path to license json file for Spark OCR
 - `-a`    path to a single license json file for both Spark OCR and Spark NLP
 - `-s`    specify pyspark version
 - `-p`    specify port of jupyter notebook

Use the `-i` flag for installing the libraries in a new virtual environment. 

You can provide the desired path for virtual env using `-v` flag, otherwise a default location of `./sparknlp_env` will be selected. 

The `PATH_TO_LICENSE_JSON_FILE` parameter must be replaced with the path where the license file is available on the local machine. According to the libraries you want to use different flags are available: `-j`, `-o` or `-a`. The license files can be easily downloaded from *My Subscription* section in your [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/) account.

To start using Jupyter Notebook after the installation of the libraries use the `-r` flag. 

The install script downloads a couple of example notebooks that you can use to start experimenting with the libraries. Those will be availabe under `./JohnSnowLabs/example_notebooks` folder. 


## Install via Docker

A docker image that contains all the required libraries for installing and running Spark NLP for Healthcare is also available. However, it does not contain the library itself, as it is licensed, and requires installation credentials.

Make sure you have a valid license for Spark NLP for Healthcare (in case you do not have one, you can ask for a trial [here](https://www.johnsnowlabs.com/install/) ), and follow the instructions below:


### Instructions

- Run the following commands to download the `docker-compose.yml` and the `sparknlp_keys.txt` files on your local machine:
```bash
curl -o docker-compose.yaml https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/513a4d682f11abc33b2e26ef8a9d72ad52a7b4f0/jupyter/docker_image_nlp_hc/docker-compose.yaml
curl -o sparknlp_keys.txt https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/docker_image_nlp_hc/sparknlp_keys.txt
```

- Download your license key in json format from [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/) 
- Populate License keys in `sparknlp_keys.txt` file.
- Run the following command to run the container in detached mode:

```bash
 docker-compose up -d
 ``` 
- By default, the jupyter notebook runs on port `8888` - you can access it by typing `localhost:8888` in your browser.


### Troubleshooting

- Make sure docker is installed on your system.
- If you face any error while importing the lib inside jupyter, make sure all the credentials are correct in the key files and restart the service again.
- If the default port `8888` is already occupied by another process, please change the mapping.
- You can change/adjust volume and port mapping in the `docker-compose.yml` file.
- You don't have a license key? Ask for a trial license [here](https://www.johnsnowlabs.com/install/).

## Install locally on Python

You can install the Spark NLP for Healthcare package by using:

```bash
pip install -q spark-nlp-jsl==${version} --extra-index-url https://pypi.johnsnowlabs.com/${secret.code} --upgrade
```

`{version}` is the version part of the `{secret.code}` (`{secret.code}.split('-')[0]`) (i.e. `2.6.0`)

The `{secret.code}` is a secret code that is only available to users with valid/trial license. 

You can ask for a free trial for Spark NLP for Healthcare [here](https://www.johnsnowlabs.com/install/). Then, you can obtain the secret code by visiting your account on [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/). Read more on how to get a license [here](licensed_install#get-a-spark-nlp-for-healthcare-license).


### Setup AWS-CLI Credentials for licensed pretrained models

Starting from Spark NLP for Healthcare version 2.4.2, you need to first setup your AWS credentials to be able to access the private repository for John Snow Labs Pretrained Models.
You can do this setup via Amazon AWS Command Line Interface (AWSCLI).

Instructions about how to install AWSCLI are available at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html">Installing the AWS CLI</a>

Make sure you configure your credentials with AWS configure following the instructions at:

<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html">Configuring the AWS CLI</a>

Please substitute the `ACCESS_KEY` and `SECRET_KEY` with the credentials available on your license json file. This is available on your account from [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/). [Read this](licensed_install#get-a-spark-nlp-for-healthcare-license) for more information.


### Start Spark NLP for Healthcare Session from Python

The following will initialize the spark session in case you have run the Jupyter Notebook directly. If you have started the notebook using
pyspark this cell is just ignored.

Initializing the spark session takes some seconds (usually less than 1 minute) as the jar from the server needs to be loaded.

The `{secret.code}` is a secret code that is only available to users with valid/trial license. 

You can ask for a free trial for Spark NLP for Healthcare [here](https://www.johnsnowlabs.com/install/). Then, you can obtain the secret code by visiting your account on [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/). Read more on how to get a license [here](licensed_install#get-a-spark-nlp-for-healthcare-license).

You can either use our convenience function to start your Spark Session that will use standard configuration arguments:

```python
import sparknlp_jsl
spark = sparknlp_jsl.start(SECRET)
```

Or use the SparkSession module for more flexibility:

```python
from pyspark.sql import SparkSession

def start(SECRET):
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:"+PUBLIC_VERSION) \
        .config("spark.jars", "https://pypi.johnsnowlabs.com/"+SECRET+"/spark-nlp-jsl-"+JSL_VERSION+".jar")
      
    return builder.getOrCreate()

spark = start(SECRET)
```

If you want to download the source files (jar and whl files) locally, you can follow the instructions <a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/SparkNLP_offline_installation.ipynb">here</a>.


### Spark NLP for Healthcare Cheat Sheet

```bash
# Install Spark NLP from PyPI
pip install spark-nlp==3.2.3

#install Spark NLP helathcare

pip install spark-nlp-jsl==${version} --extra-index-url https://pypi.johnsnowlabs.com/${secret.code} --upgrade

# Load Spark NLP with Spark Shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.3 --jars spark-nlp-jsl-${version}.jar

# Load Spark NLP with PySpark
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.3  --jars spark-nlp-jsl-${version}.jar

# Load Spark NLP with Spark Submit
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.3 --jars spark-nlp-jsl-${version}.jar
```

## Install locally for Scala

### Use Spark NLP for Healthcare in Spark shell

1.Download the fat jar for Spark NLP for Healthcare

```bash
aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl-$jsl_version.jar spark-nlp-jsl-$jsl_version.jar
```

2.Set up the `Environment Variables` box:

```bash
    AWS_ACCESS_KEY_ID=xxx
    AWS_SECRET_ACCESS_KEY=yyy
    SPARK_NLP_LICENSE=zzz
```

3.The preferred way to use the library when running Spark programs is using the `--packages`and `--jar` option as specified in the `spark-packages` section. 

```bash
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:${public-version} --jars /spark-nlp-jsl-${version}.jar
```

### Use Spark NLP for Healthcare in Sbt project

1.Download the fat jar for Spark NLP for Healthcare.
```bash
aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl-$jsl_version.jar spark-nlp-jsl-$jsl_version.jar
```

2.Set up the `Environment Variables` box:

```bash
    AWS_ACCESS_KEY_ID=xxx
    AWS_SECRET_ACCESS_KEY=yyy
    SPARK_NLP_LICENSE=zzz
```

3.Add the spark-nlp jar in your build.sbt project

```scala
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "{public-version}"
````

4.You need to create the /lib folder and paste the spark-nlp-jsl-${version}.jar file.
 
5.Add the fat spark-nlp-healthcare in your classpath. You can do it by adding this line in your build.sbt

```scala
unmanagedJars in Compile += file("lib/sparknlp-jsl.jar")
```


## Install on Databricks

### Automatic deployment of John Snow Labs NLP libraries

You can automatically deploy John Snow Labs libraries on Databricks by filling in the form available [here](https://www.johnsnowlabs.com/databricks/). 
This will allow you to start a 30-day free trial with no limit on the amount of processed data. You just need to provide a Databricks Access Token that is used by our deployment script to connect to your Databricks instance and install John Snow Labs NLP libraries on a cluster of your choice.

### Manual deployment of Spark NLP for Healthcare

1. Create a cluster if you don't have one already
2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab, in `Spark.Config` box:

    ```bash
    spark.kryoserializer.buffer.max 1000M
    spark.serializer org.apache.spark.serializer.KryoSerializer
    spark.driver.extraJavaOptions -Dspark.jsl.settings.pretrained.credentials.secret_access_key=xxx -Dspark.jsl.settings.pretrained.credentials.access_key_id=yyy

    ```
      -  Please add the following to the `Advanced Options -> Spark` tab, in `Environment Variables` box:

    ```bash
    SPARK_NLP_LICENSE=zzz
    ```
Note: Spark-NLP for Healthcare also support reading the license from the Databricks DFS, on the fixed location, dbfs:/FileStore/johnsnowlabs/license.key. 
The precedence for that location is the highest, so make sure that file is not containing any outdated license key.

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
 - Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:${version}` -> Install
 - Please add following jars:
        - Install New -> Python Whl -> upload `https://pypi.johnsnowlabs.com/${secret.code}/spark-nlp-jsl/spark_nlp_jsl-${version}-py3-none-any.whl`
        - Install New -> Jar -> upload `https://pypi.johnsnowlabs.com/${secret.code}/spark-nlp-jsl-${version}.jar`
4. Now you can attach your notebook to the cluster and use Spark NLP!

## Windows Support

In order to fully take advantage of Spark NLP on Windows (8 or 10), you need to setup/install Apache Spark, Apache Hadoop, Java and a Pyton environment correctly by following the following instructions: [https://github.com/JohnSnowLabs/spark-nlp/discussions/1022](https://github.com/JohnSnowLabs/spark-nlp/discussions/1022)


### How to correctly install Spark NLP on Windows

Follow the below steps to set up Spark NLP with Spark 3.1.2:

  1. Download [Adopt OpenJDK 1.8](https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot)
     - Make sure it is 64-bit
     - Make sure you install it in the root of your main drive `C:\java`.
     - During installation after changing the path, select setting Path

  2. Download the pre-compiled Hadoop binaries `winutils.exe`, `hadoop.dll` and put it in a folder called `C:\hadoop\bin` from [https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin](https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin)
     - **Note:** The version above is for Spark 3.1.2, which was built for Hadoop 3.2.0. You might have to change the hadoop version in the link, depending on which Spark version you are using.

  3. Download [Apache Spark 3.1.2](https://www.apache.org/dyn/closer.lua/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz) and extract it to `C:\spark`.

  4. Set/add environment variables for `HADOOP_HOME` to `C:\hadoop` and `SPARK_HOME` to `C:\spark`.

  5. Add `%HADOOP_HOME%\bin` and `%SPARK_HOME%\bin` to the `PATH` environment variable.

  6. Install [Microsoft Visual C++ 2010 Redistributed Package (x64)](https://www.microsoft.com/en-us/download/details.aspx?id=26999).

  7. Create folders `C:\tmp` and `C:\tmp\hive`
     - If you encounter issues with permissions to these folders, you might need
       to change the permissions by running the following commands:
       ```
       %HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/hive
       %HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/
       ```

#### Requisites for PySpark

We recommend using `conda` to manage your Python environment on Windows.

- Download [Miniconda for Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe)
- See [Quick Install](#quick-install) on how to set up a conda environment with
  Spark NLP.
- The following environment variables need to be set:
  - `PYSPARK_PYTHON=python`
  - Optionally, if you want to use the Jupyter Notebook runtime of Spark:
    - first install it in the environment with `conda install notebook`
    - then set `PYSPARK_DRIVER_PYTHON=jupyter`, `PYSPARK_DRIVER_PYTHON_OPTS=notebook`
  - The environment variables can either be directly set in windows, or if only
    the conda env will be used, with `conda env config vars set PYSPARK_PYTHON=python`.
    After setting the variable with conda, you need to deactivate and re-activate
    the environment.

Now you can use the downloaded binary by navigating to `%SPARK_HOME%\bin` and
running

Either create a conda env for python 3.6, install *pyspark==3.1.2 spark-nlp numpy* and use Jupyter/python console, or in the same conda env you can go to spark bin for *pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.4*.

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127225-21662880-dd65-11ea-8b98-3a2c26cfa534.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127243-2925cd00-dd65-11ea-9b20-ba3353473a98.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Use on Google Colab

Run the following code in Google Colab notebook and start using Spark NLP right away.

The first thing that you need is to create the json file with the credentials and the configuration in your local system.

```json
{
  "PUBLIC_VERSION": "3.2.3",
  "JSL_VERSION": "{version}",
  "SECRET": "{version}-{secret.code}",
  "SPARK_NLP_LICENSE": "xxxxx",
  "AWS_ACCESS_KEY_ID": "yyyy",
  "AWS_SECRET_ACCESS_KEY": "zzzz"
}
```

If you have a valid floating license, the license json file can be downloaded from your account on [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/) on **My Subscriptions** section. To get a trial license please visit 


Then you need to write that piece of code to load the credentials that you created before.

```python

import json
import os

from google.colab import files

license_keys = files.upload()

with open(list(license_keys.keys())[0]) as f:
  license_keys = json.load(f)

# Defining license key-value pairs as local variables
locals().update(license_keys)

# Adding license key-value pairs to environment variables
os.environ.update(license_keys)
```

```sh
# This is only to setup PySpark and Spark NLP on Colab
!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jsl_colab_setup.sh
```


```sh
# -p is for pyspark (by default 3.1.1)
!bash jsl_colab_setup.sh
```

[Spark NLP quick start on Google Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) is a live demo on Google Colab that performs named entity recognitions for HealthCare.




## Install on GCP Dataproc

- You can follow the steps here for [installation via IU](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/platforms/dataproc)

1. Create a cluster if you don't have one already as follows.

At gcloud shell:

```bash
gcloud services enable dataproc.googleapis.com \
  compute.googleapis.com \
  storage-component.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com
```

```bash
REGION=<region>
```

```bash
BUCKET_NAME=<bucket_name>
gsutil mb -c standard -l ${REGION} gs://${BUCKET_NAME}
```

```bash
REGION=<region>
ZONE=<zone>
CLUSTER_NAME=<cluster_name>
BUCKET_NAME=<bucket_name>
```

You can set image-version, master-machine-type, worker-machine-type,
master-boot-disk-size, worker-boot-disk-size, num-workers as your needs.
If you use the previous image-version from 2.0, you should also add ANACONDA to optional-components.
And, you should enable gateway.
As noticed below, you should explicitly write JSL_SECRET and JSL_VERSION at metadata param inside the quotes.
This will start the pip installation using the wheel file of Licensed SparkNLP!


```bash
gcloud dataproc clusters create ${CLUSTER_NAME} \
  --region=${REGION} \
  --network=${NETWORK} \
  --zone=${ZONE} \
  --image-version=2.0 \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-2 \
  --master-boot-disk-size=128GB \
  --worker-boot-disk-size=128GB \
  --num-workers=2 \
  --bucket=${BUCKET_NAME} \
  --optional-components=JUPYTER \
  --enable-component-gateway \
  --metadata 'PIP_PACKAGES=google-cloud-bigquery google-cloud-storage spark-nlp-display
  https://s3.eu-west-1.amazonaws.com/pypi.johnsnowlabs.com/JSL_SECRET/spark-nlp-jsl/spark_nlp_jsl-JSL_VERSION-py3-none-any.whl' \
  --initialization-actions gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh
```

2. On an existing one, you need to install spark-nlp and spark-nlp-display packages from PyPI.

3. Now, you can attach your notebook to the cluster and use Spark NLP via following the instructions.
The key part of this usage is how to start SparkNLP sessions using Apache Hadoop YARN cluster manager.

3.1. Read license file from the notebook using GCS.

3.2. Set the right path of the Java Home Path.

3.3. Use the start function to start the SparkNLP JSL version such as follows:

```Python
def start(secret):
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:"+PUBLIC_VERSION) \
        .config("spark.jars", "https://pypi.johnsnowlabs.com/"+SECRET+"/spark-nlp-jsl-"+JSL_VERSION+".jar")

    return builder.getOrCreate()

spark = start(SECRET)
```

As you see, we did not set `.master('local[*]')` explicitly to let YARN manage the cluster.
Or you can set `.master('yarn')`.


## Install on AWS SageMaker

1. Access AWS Sagemaker in AWS.
2. Go to Notebook -> Notebook Instances.
3. Create a new Notebook Instance, follow this [Instructions Steps](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/platforms/sagemaker)
4. Minimum requirement 16G RAM and 50G Volume. This is the configuration we have used, although most of the interesting models will require a ml.t3.xlarge instance or more. Reserve at least 50GB of memory
5. Once created, open JupyterLab and use Conda Python 3 kernel.
6. Upload `license key` and set `Environment Variables`.
```Python
import json
import os

with open('spark_nlp_for_healthcare.json', 'r') as f:
    for k, v in json.load(f).items():
        %set_env $k=$v

%set_env PYSPARK=3.2.2
%set_env SPARK_HOME=/home/ec2-user/SageMaker/spark-3.2.2-bin-hadoop2.7
```
7. Download and install libraries
```Python
!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jsl_sagemaker_setup.sh
!bash jsl_sagemaker_setup.sh
```
8. Import libraries and start session
```Python
import sparknlp
import sparknlp_jsl
from pyspark.sql import SparkSession

spark = sparknlp_jsl.start(license_keys['SECRET'])
```



## Install with Poetry


This is a sample `project.toml` file which you can use with `poetry install` to setup spark NLP + the Healthcare python library `spark-nlp-jsl`

You need to point it to either the `tar.gz` or `.whl` file which are hosted at
`https://pypi.johnsnowlabs.com/<SECRET>/spark-nlp-jsl/`

**NOTE** You must update the `url` whenever you are `upgrading` your spark-nlp-jsl version

```sh
[tool.poetry]
name = "poertry_demo"
version = "0.1.0"
description = ""
authors = ["person <person@gmail.com>"]

[tool.poetry.dependencies]
python = "^3.7"

[tool.poetry.dev-dependencies]
spark-nlp = "3.4.4"
spark-nlp-jsl = { url = "https://pypi.johnsnowlabs.com/SECRET/spark-nlp-jsl/spark_nlp_jsl-tar.gz_OR_.whl" }

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## Spark-NLP for Healthcare in AWS EMR

In this page we explain how to setup Spark-NLP + Spark-NLP Healthcare in AWS EMR, using the AWS console.

### Steps
1. You must go to the blue button "Create Cluster" on the UI. By doing that you will get directed to the "Create Cluster - Quick Options" page. Don't use the quick options, click on "Go to advanced options" instead.
2. Now in Advanced Options, on Step 1, "Software and Steps", please pick the following selection in the checkboxes,
![software config](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/platforms/emr/software_configs.png?raw=true)
Also in the "Edit Software Settings" page, enter the following,

```
[{
  "Classification": "spark-env",
  "Configurations": [{
    "Classification": "export",
    "Properties": {
      "PYSPARK_PYTHON": "/usr/bin/python3",
      "AWS_ACCESS_KEY_ID": "XYXYXYXYXYXYXYXYXYXY",
      "AWS_SECRET_ACCESS_KEY": "XYXYXYXYXYXYXYXYXYXY", 
      "SPARK_NLP_LICENSE": "XYXYXYXYXYXYXYXYXYXYXYXYXYXY"
    }
  }]
},
{
  "Classification": "spark-defaults",
    "Properties": {
      "spark.yarn.stagingDir": "hdfs:///tmp",
      "spark.yarn.preserve.staging.files": "true",
      "spark.kryoserializer.buffer.max": "2000M",
      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
      "spark.driver.maxResultSize": "0",
      "spark.driver.memory": "32G"
    }
}]
```
Make sure that you replace all the secret information(marked here as XYXYXYXYXY) by the appropriate values that you received with your license.<br/>
3. In "Step 2" choose the hardware and networking configuration you prefer, or just pick the defaults. Move to next step by clocking the "Next" blue button.<br/>
4. Now you are in "Step 3", in which you assign a name to your cluster, and you can change the location of the cluster logs. If the location of the logs is OK for you, take note of the path so you can debug potential problems by using the logs.<br/>
5. Still on "Step 3", go to the bottom of the page, and expand the "Bootstrap Actions" tab. We're gonna add an action to execute during bootstrap of the cluster. Select "Custom Action", then press on "Configure and add".<br/>
You need to provide a path to a script on S3. The path needs to be public. Keep this in mind, no secret information can be contained there.<br/>
The script we'll used for this setup is [emr_bootstrap.sh](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/platforms/emr/emr_bootstrap.sh) .
<br/>
This script will install Spark-NLP 3.1.0, and Spark-NLP Healthcare 3.1.1. You'll have to edit the script if you need different versions.<br/>
After you entered the route to S3 in which you place the `emr_bootstrap.sh` file, and before clicking "add" in the dialog box, you must pass an additional parameter containing the SECRET value you received with your license. Just paste the secret on the "Optional arguments" field in that dialog box.<br/>
6. There's not much additional setup you need to perform. So just start a notebook server, connect it to the cluster you just created(be patient, it takes a while), and test with the [NLP_EMR_Setup.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/platforms/emr/NLP_EMR_Setup.ipynb) test notebook.<br/>


## Amazon Linux 2 Support

```bash
# Update Package List & Install  Required Packages
sudo yum update
sudo yum install -y amazon-linux-extras
sudo yum -y install python3-pip

# Create Python virtual environment and activate it:
python3 -m venv .sparknlp-env
source .sparknlp-env/bin/activate
```

Check JAVA version: 
- For Sparknlp versions above 3.x, please use JAVA-11
- For Sparknlp versions below 3.x and SparkOCR, please use JAVA-8

Checking Java versions installed on your machine: 
```bash
sudo alternatives --config java
```

You can pick the index number (I am using java-8 as default - index 2):

</div><div class="h3-box" markdown="1">

<img class="image image--xl" src="/assets/images/installation/amazon-linux.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div><div class="h3-box" markdown="1">

If you dont have java-11 or java-8 in you system, you can easily install via:

```bash
sudo yum install java-1.8.0-openjdk
```

Now, we can start installing the required libraries:

```bash
pip install jupyter
```

We can start jupyter notebook via:
```bash
jupyter notebook
```

```python
### Now we are in the jupyter notebook cell:
import json
import os

with open('sparknlp_for_healthcare.json) as f:
    license_keys = json.load(f)

# Defining license key-value pairs as local variables
locals().update(license_keys)

# Adding license key-value pairs to environment variables
os.environ.update(license_keys)

# Installing pyspark and spark-nlp
! pip install --upgrade -q pyspark==3.1.2 spark-nlp==$PUBLIC_VERSION

# Installing Spark NLP Healthcare
! pip install --upgrade -q spark-nlp-jsl==$JSL_VERSION  --extra-index-url https://pypi.johnsnowlabs.com/$SECRET
```



## Get a Spark NLP for Healthcare license 

You can ask for a free trial for Spark NLP for Healthcare [here](https://www.johnsnowlabs.com/install/). This will automatically create a new account for you on [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/). Login in to your new account and from `My Subscriptions` section, you can download your license key as a json file.

The license json file contains:
- the secrets for installing the Spark NLP for Healthcare and Spark OCR libraries, 
- the license key as well as 
- AWS credentials that you need to access the s3 bucket where the healthcare models and pipelines are published.  

If you have asked for a trial license but you cannot access your account on [my.JohnSnowLabs.com](https://my.johnsnowlabs.com/) and you did not receive the license information via email, please contact us at <a href="mailto:support@johnsnowlabs.com">support@johnsnowlabs.com</a>.
