---
layout: docs
header: true
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2021-01-08"
---

## Spark NLP Cheat Sheet

```bash
# Install Spark NLP from PyPI
$pip install spark-nlp==2.7.4

# Install Spark NLP from Anacodna/Conda
conda install -c johnsnowlabs spark-nlp

# Load Spark NLP with Spark Shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4

# Load Spark NLP with PySpark
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4

# Load Spark NLP with Spark Submit
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4

# Load Spark NLP as external JAR after compiling and building Spark NLP by `sbt assembly`
spark-shell --jar spark-nlp-assembly-2.7.4
```

**NOTE**: To use Spark NLP on Apache Spark 2.3.x you should instead use the following packages:

- CPU: `com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.4`
- GPU: `com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:2.7.4`

## Python

<div class="h3-box" markdown="1">

### Quick Install

Let's create a new Conda environment to manage all the dependencies there. You can use Python Virtual Environment if you prefer or not have any enviroment.

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.6 -y
$ conda activate sparknlp
$ pip install spark-nlp==2.7.4 pyspark==2.4.7
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

### Start Spark NLP Session from python

If you need to manually start SparkSession because you have other configuraations and `sparknlp.start()` is not including them, you can manually start the SparkSession:

```python
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[4]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()
```

</div>

## Scala and Java

<div class="h3-box" markdown="1">

Our package is deployed to maven central. In order to add this package
as a dependency in your application:

**spark-nlp** on Apacahe Spark 2.4.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp** on Apacahe Spark 2.3.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

</div><div class="h3-box" markdown="1">

### SBT

**spark-nlp** on Apacahe Spark 2.4.x:

```shell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.7.4"
```

**spark-nlp-gpu:**

```shell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "2.7.4"
```

**spark-nlp** on Apacahe Spark 2.3.x:

```shell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-spark23" % "2.7.4"
```

**spark-nlp-gpu:**

```shell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu-spark23" % "2.7.4"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

</div>

## Databricks

<div class="h3-box" markdown="1">

### Databricks Support

Spark NLP 2.7.4 has been tested and is compatible with the following runtimes: 6.2, 6.2 ML, 6.3, 6.3 ML, 6.4, 6.4 ML, 6.5, 6.5 ML

</div>
<div class="h3-box" markdown="1">

### Install Spark NLP on Databricks

1. Create a cluster if you don't have one already

2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab:

```bash
spark.kryoserializer.buffer.max 1000M
spark.serializer org.apache.spark.serializer.KryoSerializer
```

3. Check `Enable autoscaling local storage` box to have persistent local storage
    
4. In `Libraries` tab inside your cluster you need to follow these steps:

    4.1. Insatll New -> PyPI -> `spark-nlp` -> Install

    4.2. Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4` -> Install

5. Now you can attach your notebook to the cluster and use Spark NLP!

</div>
<div class="h3-box" markdown="1">

### Databricks Notebooks

You can view all the Databricks notebooks from this address:

[https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html)

Note: You can import these notebooks by using their URLs.

</div>
<div class="h3-box" markdown="1">

### Windows Support

In order to fully take advantage of Spark NLP on Windows (8 or 10), you need to setup/install Apache Spark, Apache Hadoop, and Java correctly by following the following instructions: [https://github.com/JohnSnowLabs/spark-nlp/discussions/1022](https://github.com/JohnSnowLabs/spark-nlp/discussions/1022)

</div>
<div class="h3-box" markdown="1">

### How to correctly install Spark NLP on Windows 8 and 10

Follow the below steps:

  1. Download OpenJDK from here: [https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot](https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot);
      - Make sure it is 64-bit
      - Make sure you install it in the root **C:\java Windows** .
      - During installation after changing the path, select setting Path

  2. Download winutils and put it in **C:\hadoop\bin** [https://github.com/cdarlint/winutils/blob/master/hadoop-2.7.3/bin/winutils.exe](https://github.com/cdarlint/winutils/blob/master/hadoop-2.7.3/bin/winutils.exe);

  3. Download Anaconda 3.6 from Archive: [https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe);

  4. Download Apache Spark 2.4.6 and extract it in **C:\spark**

  5. Set the env for HADOOP_HOME to **C:\hadoop** and SPARK_HOME to **C:\spark**

  6. Set Paths for %HADOOP_HOME%\bin and %SPARK_HOME%\bin

  7. Install C++ [https://www.microsoft.com/en-us/download/confirmation.aspx?id=14632](https://www.microsoft.com/en-us/download/confirmation.aspx?id=14632)

  8. Create **C:\temp** and **C:\temp\hive**

  9. Fix permissions:

- C:\Users\maz>%HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/hive
- C:\Users\maz>%HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/

Either create a conda env for python 3.6, install *pyspark==2.4.6 spark-nlp numpy* and use Jupyter/python console, or in the same conda env you can go to spark bin for *pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.5*.


<div class="h3-box" markdown="1">

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127225-21662880-dd65-11ea-8b98-3a2c26cfa534.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127243-2925cd00-dd65-11ea-9b20-ba3353473a98.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div>


<div class="h3-box" markdown="1">

### How to setup Docker container with Spark NLP and PySpark

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
RUN pip3 install --no-cache-dir notebook==5.* numpy pyspark==2.4.7 spark-nlp pandas mlflow Keras scikit-spark scikit-learn scipy matplotlib pydot tensorflow==1.15.0 graphviz

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

</div>