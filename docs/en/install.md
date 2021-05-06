---
layout: docs
header: true
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2021-03-20"
---

## Spark NLP Cheat Sheet

```bash
# Install Spark NLP from PyPI
pip install spark-nlp==3.0.2

# Install Spark NLP from Anacodna/Conda
conda install -c johnsnowlabs spark-nlp

# Load Spark NLP with Spark Shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2

# Load Spark NLP with PySpark
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2

# Load Spark NLP with Spark Submit
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2

# Load Spark NLP as external JAR after compiling and building Spark NLP by `sbt assembly`
spark-shell --jar spark-nlp-assembly-3.0.2
```
<div class="h3-box" markdown="1">

## Python

Spark NLP supports Python 3.6.x and 3.7.x if you are using PySpark 2.3.x or 2.4.x and Python 3.8.x if you are using PySpark 3.x.

#### Quick Install

Let's create a new Conda environment to manage all the dependencies there. You can use Python Virtual Environment if you prefer or not have any enviroment.

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.7 -y
$ conda activate sparknlp
$ pip install spark-nlp==3.0.2 pyspark==3.1.1
```

Of course you will need to have jupyter installed in your system:

```bash
pip install jupyter
```

Now you should be ready to create a jupyter notebook running from terminal:

```bash
jupyter notebook
```

</div><div class="h3-box" markdown="1">

#### Start Spark NLP Session from python

If you need to manually start SparkSession because you have other configuraations and `sparknlp.start()` is not including them, you can manually start the SparkSession:

```python
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[4]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2")\    
    .getOrCreate()
```

</div><div class="h3-box" markdown="1">

## Scala and Java

#### Maven

Spark NLP supports Scala 2.11.x if you are using Apache Spark 2.3.x or 2.4.x and Scala 2.12.x if you are using Apache Spark 3.0.x or 3.1.x. Our packages are deployed to Maven central. To add any of our packages as a dependency in your application you can follow these coordinates:

**spark-nlp** on Apache Spark 3.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark24 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark24 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>3.0.2/version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

</div><div class="h3-box" markdown="1">

#### SBT

**spark-nlp** on Apache Spark 3.x.x:

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "3.0.2"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "3.0.2"
```

**spark-nlp** on Apache Spark 2.4.x:

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-spark24" % "3.0.2"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu-spark24" % "3.0.2"
```

**spark-nlp** on Apache Spark 2.3.x:

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-spark23" % "3.0.2"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu-spark23" % "3.0.2"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

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
!bash colab.sh -p 3.1.1 -s 3.0.2
```

[Spark NLP quick start on Google Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs named entity recognitions and sentiment analysis by using Spark NLP pretrained pipelines.

</div><div class="h3-box" markdown="1">

## Kaggle Kernel

Run the following code in Kaggle Kernel and start using spark-nlp right away.

```sh
# Let's setup Kaggle for Spark NLP and PySpark
!wget http://setup.johnsnowlabs.com/kaggle.sh -O - | bash
```

[Spark NLP quick start on Kaggle Kernel](https://www.kaggle.com/mozzie/spark-nlp-named-entity-recognition) is a live demo on Kaggle Kernel that performs named entity recognitions by using Spark NLP pretrained pipeline.

</div><div class="h3-box" markdown="1">

## Databricks Support

Spark NLP 3.0.2 has been tested and is compatible with the following runtimes:

- 5.5 LTS
- 5.5 LTS ML & GPU
- 6.4
- 6.4 ML & GPU
- 7.3
- 7.3 ML & GPU
- 7.4
- 7.4 ML & GPU
- 7.5
- 7.5 ML & GPU
- 7.6
- 7.6 ML & GPU
- 8.0
- 8.0 ML
- 8.1 Beta

NOTE: The Databricks 8.1 Beta ML with GPU is not supported in Spark NLP 3.0.2 due to its default CUDA 11.x incompatibility

</div><div class="h3-box" markdown="1">

#### Install Spark NLP on Databricks

1. Create a cluster if you don't have one already

2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab:

    ```bash
    spark.kryoserializer.buffer.max 2000M
    spark.serializer org.apache.spark.serializer.KryoSerializer
    ```

3. In `Libraries` tab inside your cluster you need to follow these steps:

    3.1. Install New -> PyPI -> `spark-nlp` -> Install

    3.2. Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2` -> Install

4. Now you can attach your notebook to the cluster and use Spark NLP!

</div><div class="h3-box" markdown="1">

#### Databricks Notebooks

You can view all the Databricks notebooks from this address:

[https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html)

Note: You can import these notebooks by using their URLs.

</div><div class="h3-box" markdown="1">

## EMR Support

Spark NLP 3.0.2 has been tested and is compatible with the following EMR releases:

- emr-5.20.0
- emr-5.21.0
- emr-5.21.1
- emr-5.22.0
- emr-5.23.0
- emr-5.24.0
- emr-5.24.1
- emr-5.25.0
- emr-5.26.0
- emr-5.27.0
- emr-5.28.0
- emr-5.29.0
- emr-5.30.0
- emr-5.30.1
- emr-5.31.0
- emr-5.32.0
- emr-6.1.0
- emr-6.2.0

Full list of [Amazon EMR 5.x releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-5x.html)
Full list of [Amazon EMR 6.x releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-6x.html)

NOTE: The EMR 6.0.0 is not supported by Spark NLP 3.0.2

</div><div class="h3-box" markdown="1">

#### How to create EMR cluster via CLI

To lanuch EMR cluster with Apache Spark/PySpark and Spark NLP correctly you need to have bootstrap and software configuration.

A sample of your bootstrap script

```.sh
#!/bin/bash
set -x -e

echo -e 'export PYSPARK_PYTHON=/usr/bin/python3 
export HADOOP_CONF_DIR=/etc/hadoop/conf 
export SPARK_JARS_DIR=/usr/lib/spark/jars 
export SPARK_HOME=/usr/lib/spark' >> $HOME/.bashrc && source $HOME/.bashrc

sudo python3 -m pip install awscli boto spark-nlp

set +x
exit 0

```

A sample of your software configuration in JSON on S3 (must be public access):

```.json
[{
  "Classification": "spark-env",
  "Configurations": [{
    "Classification": "export",
    "Properties": {
      "PYSPARK_PYTHON": "/usr/bin/python3"
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
      "spark.jars.packages": "com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2"
    }
}
]

A sample of AWS CLI to launch EMR cluster:

```.sh
aws emr create-cluster \
--name "Spark NLP 3.0.2" \
--release-label emr-6.2.0 \
--applications Name=Hadoop Name=Spark Name=Hive \
--instance-type m4.4xlarge \
--instance-count 3 \
--use-default-roles \
--log-uri "s3://<S3_BUCKET>/" \
--bootstrap-actions Path=s3://<S3_BUCKET>/emr-bootstrap.sh,Name=custome \
--configurations "https://<public_access>/sparknlp-config.json" \
--ec2-attributes KeyName=<your_ssh_key>,EmrManagedMasterSecurityGroup=<security_group_with_ssh>,EmrManagedSlaveSecurityGroup=<security_group_with_ssh> \
--profile <aws_profile_credentials>
```

</div><div class="h3-box" markdown="1">

## Docker Support

For having Spark NLP, PySpark, Jupyter, and other ML/DL dependencies as a Docker image you can use the following template:

```bash
#Download base image ubuntu 18.04
FROM ubuntu:18.04

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

RUN apt-get update && apt-get install -y \
    tar \
    wget \
    bash \
    rsync \
    gcc \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libzmq3-dev \
    python3 \ 
    python3-dev \
    python3-pip \
    unzip \
    pkg-config \
    software-properties-common \
    graphviz

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;
# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install --upgrade pip
# You only need pyspark and spark-nlp paclages to use Spark NLP
# The rest of the PyPI packages are here as examples
RUN pip3 install --no-cache-dir pyspark==3.1.1 spark-nlp==3.0.2 notebook==5.* numpy pandas mlflow Keras scikit-spark scikit-learn scipy matplotlib pydot tensorflow==2.3.1 graphviz

# Make sure the contents of our repo are in ${HOME}
RUN mkdir -p /home/jovyan/tutorials
RUN mkdir -p /home/jovyan/jupyter

COPY data ${HOME}/data
COPY jupyter ${HOME}/jupyter
COPY tutorials ${HOME}/tutorials
RUN jupyter notebook --generate-config
COPY jupyter_notebook_config.json /home/jovyan/.jupyter/jupyter_notebook_config.json
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

WORKDIR ${HOME}

# Specify the default command to run
CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
```

Finally, use **jupyter_notebook_config.json** for the password:

```bash
{
  "NotebookApp": {
    "password": "sha1:65adaa6ffb9c:36df1c2086ef294276da703667d1b8ff38f92614"
  }
}
```

</div><div class="h3-box" markdown="1">

## Windows Support

In order to fully take advantage of Spark NLP on Windows (8 or 10), you need to setup/install Apache Spark, Apache Hadoop, and Java correctly by following the following instructions: [https://github.com/JohnSnowLabs/spark-nlp/discussions/1022](https://github.com/JohnSnowLabs/spark-nlp/discussions/1022)

</div><div class="h3-box" markdown="1">

#### How to correctly install Spark NLP on Windows 8 and 10

Follow the below steps:

  1. Download OpenJDK from here: [https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot](https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot);
      - Make sure it is 64-bit
      - Make sure you install it in the root **C:\java Windows** .
      - During installation after changing the path, select setting Path

  2. Download winutils and put it in **C:\hadoop\bin** [https://github.com/cdarlint/winutils/blob/master/hadoop-2.7.3/bin/winutils.exe](https://github.com/cdarlint/winutils/blob/master/hadoop-2.7.3/bin/winutils.exe);

  3. Download Anaconda 3.6 from Archive: [https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe);

  4. Download Apache Spark 3.1.1 and extract it in **C:\spark**

  5. Set the env for HADOOP_HOME to **C:\hadoop** and SPARK_HOME to **C:\spark**

  6. Set Paths for %HADOOP_HOME%\bin and %SPARK_HOME%\bin

  7. Install C++ [https://www.microsoft.com/en-us/download/confirmation.aspx?id=14632](https://www.microsoft.com/en-us/download/confirmation.aspx?id=14632)

  8. Create **C:\temp** and **C:\temp\hive**

  9. Fix permissions:

- C:\Users\maz>%HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/hive
- C:\Users\maz>%HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/

Either create a conda env for python 3.6, install *pyspark==3.1.1 spark-nlp numpy* and use Jupyter/python console, or in the same conda env you can go to spark bin for *pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2*.

</div><div class="h3-box" markdown="1">

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127225-21662880-dd65-11ea-8b98-3a2c26cfa534.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127243-2925cd00-dd65-11ea-9b20-ba3353473a98.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div><div class="h3-box" markdown="1">

## Offline

Spark NLP library and all the pre-trained models/pipelines can be used entirely offline with no access to the Internet. If you are behind a proxy or a firewall with no access to the Maven repository (to download packages) or/and no access to S3 (to automatically download models and pipelines), you can simply follow the instructions to have Spark NLP without any limitations offline:

- Instead of using the Maven package, you need to load our Fat JAR
- Instead of using PretrainedPipeline for pretrained pipelines or the `.pretrained()` function to download pretrained models, you will need to manually download your pipeline/model from [Models Hub](https://nlp.johnsnowlabs.com/models), extract it, and load it.

Example of `SparkSession` with Fat JAR to have Spark NLP offline:

```python
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[*]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \    
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars", "/tmp/spark-nlp-assembly-3.0.2.jar")\
    .getOrCreate()
```

- You can download provided Fat JARs from each [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases), please pay attention to pick the one that suits your environment depending on the device (CPU/GPU) and Apache Spark version (2.3.x, 2.4.x, and 3.x)
- If you are local, you can load the Fat JAR from your local FileSystem, however, if you are in a cluster setup you need to put the Fat JAR on a distributed FileSystem such as HDFS, DBFS, S3, etc. (i.e., `hdfs:///tmp/spark-nlp-assembly-3.0.2.jar`)

Example of using pretrained Models and Pipelines in offline:

```python
# instead of using pretrained() for online:
# french_pos = PerceptronModel.pretrained("pos_ud_gsd", lang="fr")
# you download this model, extract it, and use .load
french_pos = PerceptronModel.load("/tmp/pos_ud_gsd_fr_2.0.2_2.4_1556531457346/")\
      .setInputCols("document", "token")\
      .setOutputCol("pos")

# example for pipelines
# instead of using PretrainedPipeline
# pipeline = PretrainedPipeline('explain_document_dl', lang='en')
# you download this pipeline, extract it, and use PipelineModel
PipelineModel.load("/tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/")
```

- Since you are downloading and loading models/pipelines manually, this means Spark NLP is not downloading the most recent and compatible models/pipelines for you. Choosing the right model/pipeline is on you
- If you are local, you can load the model/pipeline from your local FileSystem, however, if you are in a cluster setup you need to put the model/pipeline on a distributed FileSystem such as HDFS, DBFS, S3, etc. (i.e., `hdfs:///tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/`)


</div>

