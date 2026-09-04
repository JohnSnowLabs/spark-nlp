---
layout: docs
header: true
seotitle: Spark NLP - Installation
title: Spark NLP - Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2026-07-16"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## Spark NLP Cheatsheet

```bash
# Install Spark NLP from PyPI
pip install spark-nlp=={{ site.sparknlp_version }}

# Install Spark NLP from Anaconda/Conda
conda install -c johnsnowlabs spark-nlp

# Load Spark NLP with Spark Shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}

# Load Spark NLP with PySpark
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}

# Load Spark NLP with Spark Submit
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}

# Load Spark NLP as external JAR after compiling and building Spark NLP by `sbt assembly`
spark-shell --jars spark-nlp-assembly-{{ site.sparknlp_version }}.jar
```

**GPU (optional):**

Spark NLP {{ site.sparknlp_version }} is built with ONNX 1.17.0 and TensorFlow 2.7.1 deep learning engines. The minimum following NVIDIA® software are only required for GPU support:

- NVIDIA® GPU drivers version 450.80.02 or higher
- CUDA® Toolkit 11.2
- cuDNN SDK 8.1.0

</div><div class="h3-box" markdown="1">

### Spark 4 and Scala 2.13

Spark 4 uses Scala 2.13. Spark 4.0.0 has a dedicated artifact because its Spark ML `Param[T]` ABI is not binary-compatible with Spark 4.0.1 and later validated Spark 4 releases.

```bash
# Spark 4.0.0 only
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-spark400_2.13:{{ site.sparknlp_version }}

# Spark 4.0.1 and later validated Spark 4 versions
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}
```

The same naming rule applies to GPU, Apple Silicon, and Linux AArch64 artifacts, for example `spark-nlp-gpu-spark400_2.13` for Spark 4.0.0 and `spark-nlp-gpu_2.13` for the forward Spark 4 lane.

Spark 3.x with Scala 2.13 and Spark 4.x with Scala 2.12 are not supported.

## Python

Spark NLP supports Python 3.7.x and above depending on your major PySpark version.

**NOTE**: Since Spark version 3.2, Python 3.6 is deprecated. If you are using this
python version, consider sticking to lower versions of Spark.

</div><div class="h3-box" markdown="1">

#### Quick Install

Let's create a new Conda environment to manage all the dependencies there. You can use Python Virtual Environment if you prefer or not have any environment.

```bash
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.8 -y
$ conda activate sparknlp
$ pip install spark-nlp=={{ site.sparknlp_version }} pyspark==3.3.1
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

#### Start Spark NLP Session from Python

 Spark session for Spark NLP can be created (or retrieved) by using `sparknlp.start()`:

```python
import sparknlp
spark = sparknlp.start()
```

The same Python wheel supports the declared Spark 3 and Spark 4 runtime lanes. `sparknlp.start()` detects the installed PySpark version and selects the Maven artifact automatically:

| Installed PySpark | Selected artifact |
|---|---|
| Spark 3.x | `spark-nlp_2.12` |
| Spark 4.0.0 | `spark-nlp-spark400_2.13` |
| Spark 4.0.1, 4.1.0, 4.1.1, or 4.1.2 | `spark-nlp_2.13` |

The selected hardware option is applied to the same lane. For example, `sparknlp.start(gpu=True)` selects the corresponding `spark-nlp-gpu` artifact. Only one of `gpu`, `apple_silicon`, or `aarch64` may be enabled.

Migration note: the release line that introduces this mapping removes the experimental `scala213` argument from `sparknlp.start()`. Do not pass a Scala selector; install the supported PySpark runtime and let Spark NLP resolve the matching artifact.

If you need to manually start SparkSession because you have other configurations and `sparknlp.start()` is not including them,
you can manually start the SparkSession with:

```python
spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}") \
    .getOrCreate()
```

The manual example above is for Spark 3.x. For Spark 4.0.0 use `spark-nlp-spark400_2.13`; for Spark 4.0.1, 4.1.0, 4.1.1, or 4.1.2 use `spark-nlp_2.13`.

If using local jars, you can use `spark.jars` instead for comma-delimited jar files. For cluster setups, of course,
you'll have to put the jars in a reachable location for all driver and executor nodes.

</div><div class="h3-box" markdown="1">

### Python without explicit Pyspark installation

### Pip/Conda

If you installed pyspark through pip/conda, you can install `spark-nlp` through the same channel.

Pip:

```bash
pip install spark-nlp=={{ site.sparknlp_version }}
```

Conda:

```bash
conda install -c johnsnowlabs spark-nlp
```

PyPI [spark-nlp package](https://pypi.org/project/spark-nlp/) /
Anaconda [spark-nlp package](https://anaconda.org/JohnSnowLabs/spark-nlp)

Then you'll have to create a SparkSession either from Spark NLP:

```python
import sparknlp

spark = sparknlp.start()
```

**Quick example:**

```python
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# create or get Spark Session

spark = sparknlp.start()

sparknlp.version()
spark.version

# download, load and annotate a text by pre-trained pipeline

pipeline = PretrainedPipeline('recognize_entities_dl', 'en')
result = pipeline.annotate('The Mona Lisa is a 16th century oil painting created by Leonardo')
```

</div><div class="h3-box" markdown="1">

## Scala and Java

Select the Java and Scala line that matches the Spark runtime:

- Spark 3.x: Scala 2.12 and the existing Java 8/11 support baseline
- Spark 4.x: Scala 2.13 and Java 17

#### Maven

**spark-nlp** on Apache Spark 3.x

The `spark-nlp` has been published to
the [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp).

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

**spark-nlp-silicon:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-silicon -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-silicon_2.12</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

**spark-nlp-aarch64:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-aarch64 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-aarch64_2.12</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

</div><div class="h3-box" markdown="1">

#### SBT

**spark-nlp** on Apache Spark 3.x

```scala
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "{{ site.sparknlp_version }}"
```

**spark-nlp-gpu:**

```scala
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "{{ site.sparknlp_version }}"
```

**spark-nlp-silicon:**

```scala
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-silicon
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-silicon" % "{{ site.sparknlp_version }}"
```

**spark-nlp-aarch64:**

```scala
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-aarch64
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-aarch64" % "{{ site.sparknlp_version }}"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

If you are interested, there is a simple SBT project for Spark NLP to guide you on how to use it in your projects [Spark NLP SBT Starter](https://github.com/maziyarpanahi/spark-nlp-starter)

### Spark 4 / Scala 2.13 Support

Spark 4 is supported with Scala 2.13 only. Spark 3.x remains on Scala 2.12 and is not supported with Scala 2.13 in this release line.

Spark 4.0.0 requires a dedicated artifact because its Spark ML parameter ABI differs from Spark 4.0.1 and later validated Spark 4 releases.

When migrating pipelines that contain `DependencyParserModel` or `TextMatcherModel` from Spark 3/Scala 2.12 to Spark 4/Scala 2.13, export those models manually. See [Converting Spark NLP Scala 2.12 models to Scala 2.13](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/scala213/converting_models_from_212.ipynb).

| Spark runtime | CPU artifact | GPU artifact |
|---|---|---|
| Spark 4.0.0 | `spark-nlp-spark400_2.13` | `spark-nlp-gpu-spark400_2.13` |
| Spark 4.0.1, 4.1.0, 4.1.1, or 4.1.2 | `spark-nlp_2.13` | `spark-nlp-gpu_2.13` |

The same profile suffix applies to `spark-nlp-silicon` and `spark-nlp-aarch64`.

**Spark 4.0.0 Maven dependency:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark400_2.13</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

**Validated forward Spark 4 Maven dependency (4.0.1, 4.1.0, 4.1.1, and 4.1.2):**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.13</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

In Python, prefer `sparknlp.start()` so the installed PySpark version selects the correct lane automatically.

</div><div class="h3-box" markdown="1">

## Command line

Spark NLP supports all major releases of Apache Spark 3.0.x, Apache Spark 3.1.x, Apache Spark 3.2.x, Apache Spark 3.3.x, Apache Spark 3.4.x, and Apache Spark 3.5.x
This steps require internet connection.

#### Apache Spark 3.x (3.0.x, 3.1.x, 3.2.x, 3.3.x, 3.4.x, and 3.5.x - Scala 2.12)

```sh
# CPU

spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}
```

The `spark-nlp` has been published to
the [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp).

```sh
# GPU

spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{{ site.sparknlp_version }}

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{{ site.sparknlp_version }}

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{{ site.sparknlp_version }}

```

The `spark-nlp-gpu` has been published to
the [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu).

```sh
# AArch64

spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:{{ site.sparknlp_version }}

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:{{ site.sparknlp_version }}

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:{{ site.sparknlp_version }}

```

The `spark-nlp-aarch64` has been published to
the [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-aarch64).

```sh
# Apple Silicon

spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-silicon_2.12:{{ site.sparknlp_version }}

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-silicon_2.12:{{ site.sparknlp_version }}

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-silicon_2.12:{{ site.sparknlp_version }}

```

The `spark-nlp-silicon` has been published to
the [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-silicon).

**NOTE**: In case you are using large pretrained models like UniversalSentenceEncoder, you need to have the following
set in your SparkSession:

```sh
spark-shell \
  --driver-memory 16g \
  --conf spark.kryoserializer.buffer.max=2000M \
  --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}
```

</div><div class="h3-box" markdown="1">

## Installation for Apple Silicon Macs

Starting from version 4.0.0, Spark NLP has experimental support for Apple Silicon Macs.
Make sure the following prerequisites are met:

1. An Apple Silicon compatible Java version needs to be installed. We recommend [Amazon Corretto](https://docs.aws.amazon.com/corretto/latest/corretto-11-ug/downloads-list.html) Java 11, which can be easily installed with [SDKMAN!](https://sdkman.io/).

    To check if the installed Java environment is running natively on arm64, you can run the following command:

    ```shell
    johnsnow@m1mac ~ % realpath $(which java) | file -f -
    /Users/johnsnow/.sdkman/candidates/java/11.0.27-amzn/bin/java: Mach-O 64-bit executable arm64
    ```

    Note the executable type `arm64`. If it says anything else (e.g. `universal binary`, `x86_64` or `arm64e`) it might not work.

    The environment variable `JAVA_HOME` should also be set to this java version. You
    can check this by running `echo $JAVA_HOME` in your terminal. If it is not set,
    you can set it by adding `export JAVA_HOME=$(/usr/libexec/java_home)` to your
    `~/.zshrc` file.
2. If you are planning to use Annotators or Pipelines that use the RocksDB library (for
    example `WordEmbeddings`, `TextMatcher` or `explain_document_dl_en` Pipeline
    respectively) with `spark-submit`, then a workaround is required to get it working.
    See [Apple Silicon RocksDB workaround for spark-submit with Spark version >= 3.2.0](#apple-silicon-rocksdb-workaround-for-spark-submit-with-spark-version--320).

### Scala and Java Installation for Apple Silicon

Adding Spark NLP to your Scala or Java project is easy:

Simply change to dependency coordinates to `spark-nlp-silicon` and add the dependency to your
project.

How to do this is mentioned above: [Scala And Java](#scala-and-java)

So for example for Spark NLP with Apache Spark 3.0.x and 3.1.x you will end up with
maven coordinates like these:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-silicon -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-silicon_2.12</artifactId>
    <version>{{ site.sparknlp_version }}</version>
</dependency>
```

or in case of sbt:

```scala
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-silicon" % "{{ site.sparknlp_version }}"
```

If everything went well, you can now start Spark NLP with the `apple_silicon` flag set to `true`:

```scala
import com.johnsnowlabs.nlp.SparkNLP

val spark = SparkNLP.start(apple_silicon = true)
```

</div><div class="h3-box" markdown="1">

### Python for Apple Silicon

First, make sure you have a recent Python 3 installation.

```bash
johnsnow@m1mac ~ % python3 --version
Python 3.9.13
```

Then we can install the dependency as described in the [Python section](#python).
It is also recommended to use a virtual environment for this.

If everything went well, you can now start Spark NLP with the `apple_silicon` flag set to `True`:

```python
import sparknlp

spark = sparknlp.start(apple_silicon=True)
```

### Apple Silicon RocksDB workaround for spark-submit with Spark version >= 3.2.0

Starting from Spark version 3.2.0, Spark includes their own version of the RocksDB
dependency. Unfortunately, this is an older version of RocksDB does not include the
necessary binaries for Apple Silicon. To work around this issue, the default packaged RocksDB jar
has to be removed from the Spark distribution.

For example, if you downloaded Spark version 3.2.0 from the official archives, you will
find the following folders in the directory of Spark:

```bash
$ ls
bin  conf  data  examples  jars  kubernetes  LICENSE  licenses
NOTICE  python  R  README.md  RELEASE  sbin  yarn
```

To check for the RocksDB jar, you can run

```bash
$ ls jars | grep rocksdb
rocksdbjni-6.20.3.jar
```

to find the jar you have to remove. After removing the jar, the pipelines should work
as expected.

</div><div class="h3-box" markdown="1">

## Installation for Linux Aarch64 Systems

Starting from version 4.1.0, Spark NLP supports Linux systems running on an aarch64
processor architecture. The necessary dependencies have been built on Ubuntu 16.04, so a
recent system with an environment of at least that will be needed.

Check the [Python section](#python) and the [Scala And Java section](#scala-and-java) on
to install Spark NLP for your system.

</div><div class="h3-box" markdown="1">

### Starting Spark NLP

Spark NLP needs to be started with the `aarch64` flag set to `true`:

For Scala:

```scala
import com.johnsnowlabs.nlp.SparkNLP

val spark = SparkNLP.start(aarch64 = true)
```

For Python:

```python
import sparknlp

spark = sparknlp.start(aarch64=True)
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
!wget http://setup.johnsnowlabs.com/colab.sh -O - | bash /dev/stdin -p 3.4.0 -s {{ site.sparknlp_version }}
```

[Spark NLP quick start on Google Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs named entity recognitions and sentiment analysis by using Spark NLP pretrained pipelines.

</div><div class="h3-box" markdown="1">

## Kaggle Kernel

Run the following code in Kaggle Kernel and start using spark-nlp right away.

```sh
# Let's setup Kaggle for Spark NLP and PySpark
!wget http://setup.johnsnowlabs.com/kaggle.sh -O - | bash
```

[Spark NLP quick start on Kaggle Kernel](https://www.kaggle.com/mozzie/spark-nlp-named-entity-recognition) is a live demo on Kaggle Kernel that performs named entity recognitions by using Spark NLP pretrained pipeline.

</div><div class="h3-box" markdown="1">

## Apache Zeppelin

Use either one of the following options

- Add the following Maven Coordinates to the interpreter's library list

```bash
com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}
```

- Add a path to pre-built jar from [here](#compiled-jars) in the interpreter's library list making sure the jar is
  available to driver path

</div><div class="h3-box" markdown="1">

## Python in Zeppelin

Apart from the previous step, install the python module through pip

```bash
pip install spark-nlp=={{ site.sparknlp_version }}
```

Or you can install `spark-nlp` from inside Zeppelin by using Conda:

```bash
python.conda install -c johnsnowlabs spark-nlp
```

Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and
install the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as
shown earlier since it includes both scala and python side installation.

</div><div class="h3-box" markdown="1">

## Jupyter Notebook

**Recommended:**

The easiest way to get this done on Linux and macOS is to simply install `spark-nlp` and `pyspark` PyPI packages and
launch the Jupyter from the same Python environment:

```sh
$ conda create -n sparknlp python=3.8 -y
$ conda activate sparknlp
# spark-nlp by default is based on pyspark 3.x
$ pip install spark-nlp=={{ site.sparknlp_version }} pyspark==3.3.1 jupyter
$ jupyter notebook
```

Then you can use `python3` kernel to run your code with creating SparkSession via `spark = sparknlp.start()`.

**Optional:**

If you are in different operating systems and require to make Jupyter Notebook run by using pyspark, you can follow
these steps:

```bash
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}
```

Alternatively, you can mix in using `--jars` option for pyspark + `pip install spark-nlp`

If not using pyspark at all, you'll have to run the instructions
pointed [here](#python-without-explicit-pyspark-installation)

</div><div class="h3-box" markdown="1">

## Databricks Cluster

### Install Spark NLP on Databricks

1. Create a cluster if you don't have one already

2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab:

    ```bash
    spark.kryoserializer.buffer.max 2000M
    spark.serializer org.apache.spark.serializer.KryoSerializer
    ```

3. In `Libraries` tab inside your cluster you need to follow these steps:

    3.1. Install New -> PyPI -> `spark-nlp=={{ site.sparknlp_version }}` -> Install

    3.2. Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}` -> Install

4. Now you can attach your notebook to the cluster and use Spark NLP!

NOTE: Databricks' runtimes support different Apache Spark major releases. Please make sure you choose the correct Spark NLP Maven package name (Maven Coordinate) for your runtime from our [Packages Cheatsheet](https://github.com/JohnSnowLabs/spark-nlp#packages-cheatsheet)

#### ONNX GPU Inference on Databricks

To run infer ONNX models with GPU on Databricks clusters, we need to perform some additional setup steps. ONNX requires CUDA 12 and cuDNN 9 to be installed.

Therefore, we need to use Databricks runtimes starting from version 15, as these come with CUDA 12. However, they come with cuDNN 8, which we need to upgrade manually.
To do so, we have to add the following script as an [init script](https://docs.databricks.com/en/init-scripts/index.html):

```bash
#!/bin/bash
sudo apt-get update && sudo apt-get -y install cudnn9-cuda-12
```

You need to save this script to a shell script file (i.e. `upgrade-cudnn9.sh`) in your workspace. Afterwards, you need to specify it on your compute resource under the *Advanced options* section. cuDNN will be upgraded to version 9 on all nodes before Spark is started.

</div><div class="h3-box" markdown="1">

### Databricks Notebooks

You can view all the Databricks notebooks from this address:

[https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html)

Note: You can import these notebooks by using their URLs.

</div><div class="h3-box" markdown="1">

## Microsoft Fabric

Microsoft Fabric notebooks run on managed Spark 3.4 clusters, so you need to provide the Spark NLP fat JARs through OneLake/ABFSS and wire them into the runtime via Spark properties.

### Spark NLP on Microsoft Fabric

1. Inside Fabric go to a workspace and click on `+New Item` button, type `lake` on the search bar and chose `Lakehouse` and type a name for it.
   <img class="image image--xl" src="/assets/images/installation/ms-fabric-lake-house-item.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
   <img class="image image--xl" src="/assets/images/installation/ms-fabric-lake-house.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
2. Inside Fabric go to a workspace and click on `+New Item` button, type `env` on the search bar and chose `Environment` and type a name for it.
  <img class="image image--xl" src="/assets/images/installation/ms-fabric-spark-env.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
3. Choose **Fabric Runtime 1.2** (Spark 3.4 + Delta 2.4) then go to `Spark properties` and set `spark.jars`
4. Upload `spark-nlp-assembly-{{ site.sparknlp_version }}.jar` to an ABFSS folder that both driver and executors can see, for example `abfss://workspace@storage.dfs.core.windows.net/jars/`.
   <img class="image image--xl" src="/assets/images/installation/ms-fabric-spark-properties.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
5. Create a Notebook and attach it to the environment you created before.

### Spark NLP ONNX compatibility on Microsoft Fabric

Follow the steps above to set up Spark NLP, then add the following additional steps to enable ONNX inference support:

1. On `Spark properties` point `spark.executor.extraClassPath` and `spark.driver.extraClassPath` to the ABFSS jar directory to ensure ONNX classes are visible `abfss://workspace@storage.dfs.core.windows.net/jars/spark-nlp-assembly-{{ site.sparknlp_version }}.jar`.
2. On `Spark properties` enable `spark.executor.userClassPathFirst=true` and `spark.driver.userClassPathFirst=true` so the Spark NLP/ONNX classes take precedence over the Fabric runtime defaults.

These settings let Fabric distribute the Spark NLP binaries without manual copy steps and ensure ONNX inference components remain compatible with the managed runtime.

</div><div class="h3-box" markdown="1">

## EMR Cluster

To launch EMR clusters with Apache Spark/PySpark and Spark NLP correctly you need to have bootstrap and software
configuration.

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
      "spark.jars.packages": "com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}"
    }
}]
```

A sample of AWS CLI to launch EMR cluster:

```.sh
aws emr create-cluster \
--name "Spark NLP {{ site.sparknlp_version }}" \
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

## EMR Serverless

This section explains how to run Spark NLP Open Source on Amazon EMR Serverless with either Apache Spark 3.x or Apache Spark 4.x. EMR Serverless does not run bootstrap actions, so the Python environment and Spark NLP JVM artifacts must be available when the job starts.

### 1. Select the EMR, Spark, and Scala lane

Choose the Spark NLP JVM artifact that matches the Spark and Scala versions supplied by the EMR Serverless release.

{:.table-model-big}
| EMR Serverless release | Spark | Scala | Default Java | Spark NLP JVM artifact |
|---|---|---|---|---|
| `emr-7.12.0` | Spark 3.x | 2.12 | EMR-managed | `spark-nlp_2.12` |
| `emr-spark-8.0.0` | `4.0.2-amzn-0` | 2.13 | 17 | `spark-nlp_2.13` |

The Java runtime is supplied by EMR, not selected by Spark NLP. Verify the exact Java version for the chosen EMR release when using a different release label.

`emr-spark-8.0.0` is the Spark-focused Amazon EMR 8 release label; it is not `emr-8.0.0`. Spark 4 requires Scala 2.13, so do not use a `_2.12` Spark NLP artifact or other Scala 2.12 dependencies with this release. The `spark-nlp-spark400_2.13` artifact is only for Spark 4.0.0; EMR Spark 8.0.0 supplies Spark 4.0.2 and uses `spark-nlp_2.13`.

The same Spark NLP Python package is used with both Spark lines. The JVM artifact is what changes:

{:.table-model-big}
| Spark line | Maven coordinate |
|---|---|
| Spark 3.x / Scala 2.12 | `com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}` |
| Spark 4.0.2 / Scala 2.13 | `com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}` |

AWS documents the Spark 4 runtime details in [Amazon EMR Spark 8.0.0](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark800-release.html).

### 2. Create the EMR Serverless application

Create an application with the release label for the Spark line you intend to run:

```bash
export AWS_REGION='<aws-region>'
export AWS_DEFAULT_REGION="$AWS_REGION"

# Spark 3.x example:
export EMR_RELEASE_LABEL='emr-7.12.0'
export APP_NAME='spark-nlp-spark3'

# Spark 4.x example; use these values instead of the Spark 3.x values above:
# export EMR_RELEASE_LABEL='emr-spark-8.0.0'
# export APP_NAME='spark-nlp-spark4'

export APP_ID="$(aws emr-serverless create-application \
  --region "$AWS_REGION" \
  --name "$APP_NAME" \
  --type SPARK \
  --release-label "$EMR_RELEASE_LABEL" \
  --query applicationId \
  --output text)"

printf 'APP_ID=%s\n' "$APP_ID"
```

Verify the application before submitting a job:

```bash
aws emr-serverless get-application \
  --region "$AWS_REGION" \
  --application-id "$APP_ID" \
  --query 'application.{Name:name,Id:applicationId,State:state,Release:releaseLabel}' \
  --output table
```

AWS allows an application's `releaseLabel` to be updated while the application is in the `CREATED` or `STOPPED` state. A separate application is preferable when Spark 3 and Spark 4 workloads must remain independently available or validated.

The identity that submits the job and the IAM role used by the running job are different:

- the submitter creates applications and calls `StartJobRun`;
- the stable runtime role is passed through `--execution-role-arn` and accesses S3 for the job.

Do not pass a temporary `arn:aws:sts::...:assumed-role/...` submitter ARN as the execution role.

Before the first job on a new application, inspect the runtime role's trust policy:

```bash
export EXECUTION_ROLE_ARN='<emr-serverless-runtime-role-arn>'
export RUNTIME_ROLE_NAME="${EXECUTION_ROLE_ARN##*/}"

aws iam get-role \
  --role-name "$RUNTIME_ROLE_NAME" \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json
```

The trust policy must allow `emr-serverless.amazonaws.com`. If an `aws:SourceArn` condition is restricted to application ARNs, preserve every application that must continue working and add the new application ARN exactly. Do not broaden an exact allowlist to `applications/*` unless that is an explicit IAM decision.

The runtime role needs least-privilege S3 access for the job script, Python environment, JARs, inputs, outputs, and logs. See [Job runtime roles for Amazon EMR Serverless](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/security-iam-runtime-role.html).

### 3. Prepare the Spark NLP artifacts

Use separate S3 prefixes for the Spark 3 / Scala 2.12 and Spark 4 / Scala 2.13 JVM artifacts:

```text
s3://<artifact-bucket>/spark-nlp-emr/
  scripts/
  envs/
    spark-nlp-{{ site.sparknlp_version }}-py311.tar.gz
  jars/
    spark3-scala212/
    spark4-scala213/
  manifests/
  models/
  outputs/
  logs/
```

There are two dependency-delivery modes:

1. If the job has Maven Central access, pass the matching Maven coordinate with `--packages`.
2. If the job has no Maven/PyPI access, resolve the matching runtime dependency graph in a connected build environment, verify it, stage it in S3, and pass the complete comma-separated S3 JAR list through `spark.jars`.

For offline resolution, select the artifact for the application lane in a standard Maven `pom.xml`. This example is for Spark 4.x; use `spark-nlp_2.12` for Spark 3.x:

```xml
<dependency>
  <groupId>com.johnsnowlabs.nlp</groupId>
  <artifactId>spark-nlp_2.13</artifactId>
  <version>{{ site.sparknlp_version }}</version>
  <exclusions>
    <exclusion>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-library</artifactId>
    </exclusion>
    <exclusion>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-reflect</artifactId>
    </exclusion>
  </exclusions>
</dependency>
```

Resolve into a new directory so stale dependencies from an earlier build cannot be included:

```bash
export RUNTIME_JARS_DIR="$PWD/runtime-jars"
if [[ -e "$RUNTIME_JARS_DIR" ]]; then
  echo "Refusing to reuse existing directory: $RUNTIME_JARS_DIR" >&2
  exit 1
fi

mvn --batch-mode \
  -f pom.xml \
  dependency:copy-dependencies \
  -DincludeScope=runtime \
  -DoutputDirectory="$RUNTIME_JARS_DIR"
```

Do not stage Spark or Scala runtime JARs already supplied by EMR. Reject unexpected `spark-core`, `spark-sql`, `spark-mllib`, `spark-catalyst`, `spark-network-*`, `spark-launcher`, `scala-library`, and `scala-reflect` artifacts before uploading the dependency set.

Keep a SHA-256 manifest and use a new S3 prefix for each verified dependency set. Construct `SPARK_NLP_JARS` from the complete sorted list of uploaded JAR object URIs, excluding the manifest itself.

### 4. Build the Python environment

Build the environment against the operating system and Python version supplied by the selected EMR release. The following Amazon Linux 2023 and Python 3.11 example is suitable for `emr-spark-8.0.0`.

EMR supplies PySpark. Do not install or package a second PySpark distribution.

```dockerfile
FROM amazonlinux:2023

RUN dnf update -y && \
    dnf install -y \
      python3.11 \
      python3.11-pip \
      python3.11-devel \
      tar gzip findutils shadow-utils && \
    dnf clean all

WORKDIR /work
CMD ["/bin/bash"]
```

Build and enter the image:

```bash
docker build -t spark-nlp-emr-al2023 -f Dockerfile .
docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -v "$PWD":/work \
  spark-nlp-emr-al2023
```

Inside the container:

```bash
cd /work
export SPARK_NLP_VERSION='{{ site.sparknlp_version }}'

python3.11 -m venv --copies spark-nlp-env
source spark-nlp-env/bin/activate
python -m pip install --upgrade pip
python -m pip install "spark-nlp==$SPARK_NLP_VERSION" "numpy==1.26.4" venv-pack

python -c 'from importlib.metadata import version; print(version("spark-nlp"))'
if python -c 'import pyspark' 2>/dev/null; then
  echo 'The packaged environment must not contain PySpark' >&2
  exit 1
fi

venv-pack -f -o "spark-nlp-${SPARK_NLP_VERSION}-py311.tar.gz"
```

Upload the archive:

```bash
export SPARK_NLP_VERSION='{{ site.sparknlp_version }}'
aws s3 cp "spark-nlp-${SPARK_NLP_VERSION}-py311.tar.gz" \
  "s3://<artifact-bucket>/spark-nlp-emr/envs/spark-nlp-${SPARK_NLP_VERSION}-py311.tar.gz"
```

### 5. Create a Spark NLP job

The following model-free distributed smoke test works with both Spark 3.x and Spark 4.x when the correct application and JVM artifact lane are selected:

```python
from importlib.metadata import version

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import Tokenizer
from sparknlp.base import DocumentAssembler, Finisher

spark = SparkSession.builder.appName("spark-nlp-emr-serverless").getOrCreate()

print("Spark NLP Python version:", version("spark-nlp"))
print("Spark version:", spark.version)
print(
    "Scala version:",
    spark.sparkContext._jvm.scala.util.Properties.versionString(),
)
print(
    "Java version:",
    spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
)

data = spark.createDataFrame(
    [(1, "Spark NLP runs on Amazon EMR Serverless.")],
    ["id", "text"],
).repartition(2)

document = (
    DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
)
token = (
    Tokenizer()
    .setInputCols(["document"])
    .setOutputCol("token")
)
finisher = (
    Finisher()
    .setInputCols(["token"])
    .setOutputCols(["finished_token"])
)

model = Pipeline(stages=[document, token, finisher]).fit(data)
rows = model.transform(data).select("finished_token").collect()
assert "Spark" in rows[0].finished_token
print(rows)

spark.stop()
```

Save the script as `spark_nlp_emr_serverless.py`, validate its syntax, and upload it:

```bash
python3 -m py_compile spark_nlp_emr_serverless.py
aws s3 cp spark_nlp_emr_serverless.py \
  s3://<artifact-bucket>/spark-nlp-emr/scripts/spark_nlp_emr_serverless.py
```

### 6. Configure the required Spark properties

The required Spark properties for the packaged Python environment and S3-staged JAR mode are:

{:.table-model-big}
| Property | Requirement | Purpose |
|----------|-------------|---------|
| `spark.archives` | Required | Extracts the Python environment as `./environment`. Additional archives can be listed when a job needs localized resources. |
| `spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON` | Required | Forces the driver to use the packaged Python interpreter. |
| `spark.emr-serverless.driverEnv.PYSPARK_PYTHON` | Required | Sets the driver-side PySpark Python interpreter. |
| `spark.executorEnv.PYSPARK_PYTHON` | Required | Sets the executor-side PySpark Python interpreter. |
| `spark.jars` | Required for offline mode | Loads the complete Spark NLP JVM dependency set from S3 without resolving Maven packages when the job starts. Use the Spark/Scala lane that matches the application. |
| `--packages` | Alternative to `spark.jars` | Resolves the matching Spark NLP Maven coordinate when the job has Maven Central access. Do not configure both delivery modes for the same artifact. |
| `spark.serializer` | Recommended | Uses `org.apache.spark.serializer.KryoSerializer`. |
| `spark.kryoserializer.buffer.max` | Recommended | Increases the maximum Kryo buffer for larger Spark NLP annotations and models. |
| `spark.jsl.settings.pretrained.cache_folder` | Conditional | Configures a shared S3 location for pretrained resources when the job uses `.pretrained(...)` or `PretrainedPipeline(...)`. |
| `spark.hadoop.fs.s3a.impl` | Conditional | Explicitly selects Hadoop S3A when the job reads or writes `s3a://` paths and the runtime does not already configure it. |
| `spark.hadoop.fs.s3a.endpoint` | Conditional | Selects the regional S3 endpoint when required by the bucket or network configuration. |
| `spark.hadoop.fs.s3a.aws.credentials.provider` | Conditional | Selects an S3A credentials provider. Normally omit this and use the EMR Serverless runtime role. |
| `spark.jsl.settings.aws.region` | Conditional | Sets the AWS region used by Spark NLP cloud-cache operations. |
| `spark.hadoop.hive.metastore.client.factory.class` | Optional | Enables AWS Glue Data Catalog integration when the job needs Glue-backed Hive metadata. |

Do not place AWS access keys, secret keys, or session tokens in reusable Spark submit strings or public documentation. Grant the EMR Serverless runtime role access to the required S3 prefixes instead.

### 7. Submit the Spark job

The submission structure is the same for Spark 3.x and Spark 4.x. Select the application ID and runtime-JAR list for the intended lane:

{:.table-model-big}
| Spark line | Application release | Spark NLP JAR lane |
|---|---|---|
| Spark 3.x | `emr-7.12.0` | `_2.12` dependency set |
| Spark 4.x | `emr-spark-8.0.0` | `_2.13` dependency set |

Prepare the submission values. The `py311` archive name assumes that the selected EMR release uses a compatible Python 3.11 runtime; otherwise build and reference an archive for that release's Python version:

```bash
export AWS_REGION='<aws-region>'
export AWS_DEFAULT_REGION="$AWS_REGION"
export APP_ID='<application-id-for-the-selected-spark-line>'
export EXECUTION_ROLE_ARN='<emr-serverless-runtime-role-arn>'
export ENTRY_POINT='s3://<artifact-bucket>/spark-nlp-emr/scripts/spark_nlp_emr_serverless.py'
export LOG_URI='s3://<artifact-bucket>/spark-nlp-emr/logs/'
export PYTHON_ARCHIVE='s3://<artifact-bucket>/spark-nlp-emr/envs/spark-nlp-{{ site.sparknlp_version }}-py311.tar.gz'
export SPARK_NLP_JARS='<comma-separated-s3-runtime-jar-uris-for-the-selected-lane>'

: "${AWS_REGION:?AWS_REGION is required}"
: "${APP_ID:?APP_ID is required}"
: "${EXECUTION_ROLE_ARN:?EXECUTION_ROLE_ARN is required}"
: "${ENTRY_POINT:?ENTRY_POINT is required}"
: "${LOG_URI:?LOG_URI is required}"
: "${PYTHON_ARCHIVE:?PYTHON_ARCHIVE is required}"
```

Build the properties shared by both dependency-delivery modes, then add the offline S3-staged JAR list:

```bash
: "${SPARK_NLP_JARS:?SPARK_NLP_JARS is required for offline mode}"

COMMON_SPARK_SUBMIT_PARAMETERS="--conf spark.archives=${PYTHON_ARCHIVE}#environment \
--conf spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON=./environment/bin/python \
--conf spark.emr-serverless.driverEnv.PYSPARK_PYTHON=./environment/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.kryoserializer.buffer.max=2000M"

SPARK_SUBMIT_PARAMETERS="${COMMON_SPARK_SUBMIT_PARAMETERS} \
--conf spark.jars=${SPARK_NLP_JARS}"
```

When Maven Central is reachable, build the submit parameters from the same common properties and add the coordinate for the selected lane instead:

```bash
export SPARK_NLP_ARTIFACT='spark-nlp_2.12'  # Spark 3.x
# export SPARK_NLP_ARTIFACT='spark-nlp_2.13'  # Spark 4.x

SPARK_SUBMIT_PARAMETERS="${COMMON_SPARK_SUBMIT_PARAMETERS} \
--packages com.johnsnowlabs.nlp:${SPARK_NLP_ARTIFACT}:{{ site.sparknlp_version }}"
```

Do not combine that `--packages` example with an existing `spark.jars` value for Spark NLP. Start from the common Python and serializer properties, then choose exactly one JVM artifact-delivery mode.

Submit the job with JSON generated by `jq`:

```bash
JOB_DRIVER_JSON="$(jq -n \
  --arg entryPoint "$ENTRY_POINT" \
  --arg params "$SPARK_SUBMIT_PARAMETERS" \
  '{sparkSubmit:{entryPoint:$entryPoint,sparkSubmitParameters:$params}}')"

CONFIG_OVERRIDES_JSON="$(jq -n \
  --arg logUri "$LOG_URI" \
  '{monitoringConfiguration:{s3MonitoringConfiguration:{logUri:$logUri}}}')"

JOB_RUN_ID="$(aws emr-serverless start-job-run \
  --region "$AWS_REGION" \
  --application-id "$APP_ID" \
  --execution-role-arn "$EXECUTION_ROLE_ARN" \
  --name spark-nlp-serverless \
  --job-driver "$JOB_DRIVER_JSON" \
  --configuration-overrides "$CONFIG_OVERRIDES_JSON" \
  --query jobRunId \
  --output text)"

printf 'JOB_RUN_ID=%s\n' "$JOB_RUN_ID"
```

AWS documents these job-driver and configuration fields in [Using Spark configurations when you run EMR Serverless jobs](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/jobs-spark.html).

</div><div class="h3-box" markdown="1">

## GCP Dataproc

Spark NLP `{{ site.sparknlp_version }}` on Dataproc Spark 4.x uses the Spark 4 / Scala 2.13 artifact lane.

| Dataproc target | Spark line | Scala | Spark NLP artifact |
|---|---|---|---|
| Dataproc cluster `--image-version=3.0` | Spark 4.x | 2.13 | `spark-nlp_2.13` for Spark 4.0.1, 4.1.0, 4.1.1, and 4.1.2 |
| Legacy Dataproc cluster `--image-version=2.0` | Spark 3.x (image 2.0 provides Spark 3.1.3) | 2.12 | `spark-nlp_2.12` |
| Dataproc Serverless `--version=3.0` | Google-managed Spark 4 patch | 2.13 | `spark-nlp_2.13` when the runtime reports Spark 4.0.1 or a later validated Spark 4 version |
| Dataproc environment that reports Spark `4.0.0` | Spark 4.0.0 only | 2.13 | `spark-nlp-spark400_2.13` |

Notes:

- Dataproc Serverless runtime `3.0` is the Spark 4 line. Current runtime releases use Spark 4.0.1, Scala 2.13, Java 21, and Python 3.12.
- Google documents that Dataproc Serverless runtime subminor pinning is not supported. Check `spark.version` in the driver output instead of assuming that `--version=3.0` selects a specific Apache Spark patch.
- For the legacy Spark 3.x / Scala 2.12 cluster lane, use `--image-version=2.0` with `spark-nlp_2.12`. Google currently lists image 2.0 as unsupported, so verify that it remains available in your region before creating a new cluster.

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
DATAPROC_IMAGE_VERSION=3.0
SPARK_NLP_ARTIFACT=spark-nlp_2.13
# For Spark 4.0.0 only, use:
# SPARK_NLP_ARTIFACT=spark-nlp-spark400_2.13
# For the legacy Spark 3.x / Scala 2.12 lane, use:
# DATAPROC_IMAGE_VERSION=2.0
# SPARK_NLP_ARTIFACT=spark-nlp_2.12
```

For Spark 4.x, use Dataproc cluster image `3.0`. You can still tune image version subrelease, master-machine-type, worker-machine-type, master-boot-disk-size, worker-boot-disk-size, and num-workers for your workload. Enable the component gateway and set the Spark NLP Maven coordinate explicitly in the cluster properties.

```bash
gcloud dataproc clusters create ${CLUSTER_NAME} \
  --region=${REGION} \
  --zone=${ZONE} \
  --image-version=${DATAPROC_IMAGE_VERSION} \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-2 \
  --master-boot-disk-size=128GB \
  --worker-boot-disk-size=128GB \
  --num-workers=2 \
  --bucket=${BUCKET_NAME} \
  --optional-components=JUPYTER \
  --enable-component-gateway \
  --metadata 'PIP_PACKAGES=spark-nlp=={{ site.sparknlp_version }},spark-nlp-display,google-cloud-bigquery,google-cloud-storage' \
  --initialization-actions gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh \
  --properties spark:spark.serializer=org.apache.spark.serializer.KryoSerializer,spark:spark.driver.maxResultSize=0,spark:spark.kryoserializer.buffer.max=2000M,spark:spark.jars.packages=com.johnsnowlabs.nlp:${SPARK_NLP_ARTIFACT}:{{ site.sparknlp_version }}
```

2. On an existing cluster, install the `spark-nlp` and `spark-nlp-display` packages from PyPI.

3. Attach your notebook to the cluster and use Spark NLP.

For Dataproc Serverless, use `gcloud dataproc batches submit pyspark --version=3.0` for the Spark 4 line and distribute both the Spark NLP Python package and the matching Spark NLP jar lane (`spark-nlp-spark400_2.13` for Spark 4.0.0, `spark-nlp_2.13` for Spark 4.0.1 and later validated Spark 4 versions).

### Dataproc Serverless Spark 4 batch

Upload the Spark NLP `7.0.0` wheel and the assembly JAR built for the Spark runtime reported by Dataproc. The assembly filename does not encode the Maven lane, so verify that the uploaded JAR comes from the `spark-nlp_2.13` build for current runtime `3.0` releases.

```bash
PROJECT_ID=<project-id>
REGION=<region>
BUCKET_NAME=<bucket-name>
BATCH_ID=spark-nlp-700-spark4
JOB_URI=gs://${BUCKET_NAME}/python/job.py
JAR_URI=gs://${BUCKET_NAME}/jars/spark-nlp-assembly-7.0.0.jar
WHEEL_URI=gs://${BUCKET_NAME}/wheels/spark_nlp-7.0.0-py2.py3-none-any.whl

gcloud dataproc batches submit pyspark ${JOB_URI} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --batch=${BATCH_ID} \
  --version=3.0 \
  --jars=${JAR_URI} \
  --py-files=${WHEEL_URI} \
  --deps-bucket=gs://${BUCKET_NAME} \
  --properties=spark.serializer=org.apache.spark.serializer.KryoSerializer,spark.kryoserializer.buffer.max=2000M,spark.jsl.settings.gcp.project_id=${PROJECT_ID},spark.jsl.settings.pretrained.cache_folder=gs://${BUCKET_NAME}/models
```

When the launcher already supplies Spark NLP with `--jars`, let Dataproc own the Spark session instead of resolving a second Maven artifact from the Python entrypoint:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Spark NLP Dataproc Serverless").getOrCreate()
print("Spark version:", spark.version)
print("Scala version:", spark.sparkContext._jvm.scala.util.Properties.versionString())
print("Java version:", spark.sparkContext._jvm.java.lang.System.getProperty("java.version"))
```

Python `print()` and DataFrame `show()` output is written to the Dataproc batch driver output in Cloud Logging (`dataproc.googleapis.com/output`); it is not streamed into the submitting terminal.

The example uses the standard resource tier. If you explicitly select premium resources on runtime `3.0`, set `dataproc.tier=premium`. Do not combine it with the legacy `spark.dataproc.executor.compute.tier=premium` property.

References:

- [Managed Service for Apache Spark cluster image versions](https://docs.cloud.google.com/managed-spark/docs/concepts/versioning/image-version-lists)
- [Managed Service for Apache Spark runtime 3.0](https://docs.cloud.google.com/managed-spark/docs/concepts/versions/spark-runtime-3.0)
- [Managed Service for Apache Spark serverless runtime versions](https://docs.cloud.google.com/managed-spark/docs/concepts/versions/serverless-versions)
- [Monitor and troubleshoot batch workloads](https://docs.cloud.google.com/managed-spark/docs/guides/monitor-troubleshoot-batches)

## Apache Spark Support

Spark NLP supports Spark 3.x with Scala 2.12 and the following Spark 4/Scala 2.13 lanes:

| Spark runtime | Scala | Java | Maven artifact |
|---|---:|---:|---|
| Spark 4.0.0 | 2.13 | 17 | `spark-nlp-spark400_2.13` |
| Spark 4.0.1, 4.1.0, 4.1.1, or 4.1.2 | 2.13 | 17 | `spark-nlp_2.13` |

Spark 3.x with Scala 2.13 and Spark 4.x with Scala 2.12 are not supported. The table below records historical Spark NLP 4.x and 5.x compatibility.

{:.table-model-big}

| Spark NLP | Apache Spark 3.5.x | Apache Spark 3.4.x | Apache Spark 3.3.x | Apache Spark 3.2.x | Apache Spark 3.1.x | Apache Spark 3.0.x | Apache Spark 2.4.x | Apache Spark 2.3.x |
| --------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| 5.4.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.3.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.2.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.1.x     | Partially          | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.0.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 4.4.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 4.3.x     | NO                 | NO                 | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 4.2.x     | NO                 | NO                 | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 4.1.x     | NO                 | NO                 | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 4.0.x     | NO                 | NO                 | YES                | YES                | YES                | YES                | NO                 | NO                 |

Find out more about `Spark NLP` versions from our [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases).

## Historical Scala and Python Support

The current Spark 3 line uses Scala 2.12 and the Spark 4 line uses Scala 2.13. Supported Python versions follow the selected Apache Spark runtime. The table below records historical Spark NLP 4.x and 5.x compatibility.

{:.table-model-big}

| Spark NLP | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 | Scala 2.11 | Scala 2.12 |
| --------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- | ---------- |
| 5.3.x     | NO         | YES        | YES        | YES        | YES         | NO         | YES        |
| 5.2.x     | NO         | YES        | YES        | YES        | YES         | NO         | YES        |
| 5.1.x     | NO         | YES        | YES        | YES        | YES         | NO         | YES        |
| 5.0.x     | NO         | YES        | YES        | YES        | YES         | NO         | YES        |
| 4.4.x     | NO         | YES        | YES        | YES        | YES         | NO         | YES        |
| 4.3.x     | YES        | YES        | YES        | YES        | YES         | NO         | YES        |
| 4.2.x     | YES        | YES        | YES        | YES        | YES         | NO         | YES        |
| 4.1.x     | YES        | YES        | YES        | YES        | NO          | NO         | YES        |
| 4.0.x     | YES        | YES        | YES        | YES        | NO          | NO         | YES        |

## Databricks Support

Spark NLP {{ site.sparknlp_version }} has been tested and is compatible with the following runtimes:

{:.table-model-big}

|   CPU              |   GPU              |
|--------------------|--------------------|
| 9.1 / 9.1 ML       | 9.1 ML & GPU       |
| 10.1 / 10.1 ML     | 10.1 ML & GPU      |
| 10.2 / 10.2 ML     | 10.2 ML & GPU      |
| 10.3 / 10.3 ML     | 10.3 ML & GPU      |
| 10.4 / 10.4 ML     | 10.4 ML & GPU      |
| 10.5 / 10.5 ML     | 10.5 ML & GPU      |
| 11.0 / 11.0 ML     | 11.0 ML & GPU      |
| 11.1 / 11.1 ML     | 11.1 ML & GPU      |
| 11.2 / 11.2 ML     | 11.2 ML & GPU      |
| 11.3 / 11.3 ML     | 11.3 ML & GPU      |
| 12.0 / 12.0 ML     | 12.0 ML & GPU      |
| 12.1 / 12.1 ML     | 12.1 ML & GPU      |
| 12.2 / 12.2 ML     | 12.2 ML & GPU      |
| 13.0 / 13.0 ML     | 13.0 ML & GPU      |
| 13.1 / 13.1 ML     | 13.1 ML & GPU      |
| 13.2 / 13.2 ML     | 13.2 ML & GPU      |
| 13.3 / 13.3 ML     | 13.3 ML & GPU      |
| 14.0 / 14.0 ML     | 14.0 ML & GPU      |
| 14.1 / 14.1 ML     | 14.1 ML & GPU      |
| 15.x / 15.x ML     | 15.x ML & GPU      |
| 16.4 / 16.4 ML     | 16.4 ML & GPU      |

</div><div class="h3-box" markdown="1">

## EMR Support

Spark NLP {{ site.sparknlp_version }} has been tested and is compatible with the following EMR releases:

- emr-6.2.0
- emr-6.3.0
- emr-6.3.1
- emr-6.4.1
- emr-6.5.0
- emr-6.6.0
- emr-6.7.0
- emr-6.8.0
- emr-6.9.0
- emr-6.10.0
- emr-6.11.0
- emr-6.12.0
- emr-6.13.0
- emr-6.14.0

Full list of [Amazon EMR 6.x releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-6x.html)

NOTE: The EMR 6.1.0 and 6.1.1 are not supported.

</div><div class="h3-box" markdown="1">
#### How to create EMR cluster via CLI

To lanuch EMR cluster with Apache Spark/PySpark and Spark NLP correctly you need to have bootstrap and software configuration.

A sample of your bootstrap script

```sh
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

```json
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
      "spark.jars.packages": "com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}"
    }
}
]
```

A sample of AWS CLI to launch EMR cluster:

```sh
aws emr create-cluster \
--name "Spark NLP {{ site.sparknlp_version }}" \
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

Checking Java versions installed on your machine:

```bash
sudo alternatives --config java
```

You can pick the index number (I am using java-8 as default - index 2):

<img class="image image--xl" src="/assets/images/installation/amazon-linux.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

If you dont have java-11 or java-8 in you system, you can easily install via:

```bash
sudo yum install java-1.8.0-openjdk
```

Now, we can start installing the required libraries:

```bash
pip install pyspark==3.3.1
pip install spark-nlp
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
RUN pip3 install --no-cache-dir pyspark spark-nlp==3.2.3 notebook==5.* numpy pandas mlflow Keras scikit-spark scikit-learn scipy matplotlib pydot tensorflow==2.4.1 graphviz

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
    "password": "<sha1-password-hash-generated-by-jupyter>"
  }
}
```

</div><div class="h3-box" markdown="1">

## Windows Support

In order to fully take advantage of Spark NLP on Windows (8 or 10), you need to setup/install Apache Spark, Apache Hadoop, Java and a Pyton environment correctly by following the following instructions: [https://github.com/JohnSnowLabs/spark-nlp/discussions/1022](https://github.com/JohnSnowLabs/spark-nlp/discussions/1022)

</div><div class="h3-box" markdown="1">

### How to correctly install Spark NLP on Windows

Follow the below steps to set up Spark NLP with Spark 3.2.3:

  1. Download [Adopt OpenJDK 1.8](https://adoptopenjdk.net/?variant=openjdk8&jvmVariant=hotspot)
     - Make sure it is 64-bit
     - Make sure you install it in the root of your main drive `C:\java`.
     - During installation after changing the path, select setting Path

  2. Download the pre-compiled Hadoop binaries `winutils.exe`, `hadoop.dll` and put it in a folder called `C:\hadoop\bin` from [https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin](https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin)
     - **Note:** The version above is for Spark 3.2.3, which was built for Hadoop 3.2.0. You might have to change the hadoop version in the link, depending on which Spark version you are using.

  3. Download [Apache Spark 3.2.3](https://www.apache.org/dyn/closer.lua/spark/spark-3.2.3/spark-3.2.3-bin-hadoop3.2.tgz) and extract it to `C:\spark`.

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

</div><div class="h3-box" markdown="1">

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

Either create a conda env for python 3.6, install *pyspark==3.3.1 spark-nlp numpy* and use Jupyter/python console, or in the same conda env you can go to spark bin for *pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:{{ site.sparknlp_version }}*.

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127225-21662880-dd65-11ea-8b98-3a2c26cfa534.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90127243-2925cd00-dd65-11ea-9b20-ba3353473a98.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/installation/90126972-c03e5500-dd64-11ea-8285-e4f76aa9e543.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div><div class="h3-box" markdown="1">

## Offline

Spark NLP library and all the pre-trained models/pipelines can be used entirely offline with no access to the Internet. If you are behind a proxy or a firewall with no access to the Maven repository (to download packages) or/and no access to S3 (to automatically download models and pipelines), you can simply follow the instructions to have Spark NLP without any limitations offline:

- Instead of using the Maven package, you need to load our Fat JAR
- Instead of using PretrainedPipeline for pretrained pipelines or the `.pretrained()` function to download pretrained models, you will need to manually download your pipeline/model from [Models Hub](https://sparknlp.org/models), extract it, and load it.

Example of `SparkSession` with Fat JAR to have Spark NLP offline:

```python
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[*]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars", "/tmp/spark-nlp-assembly-{{ site.sparknlp_version }}.jar")\
    .getOrCreate()
```

- You can download provided Fat JARs from each [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases), please pay attention to pick the one that suits your environment depending on the device (CPU/GPU) and Apache Spark version (3.x)
- If you are local, you can load the Fat JAR from your local FileSystem, however, if you are in a cluster setup you need to put the Fat JAR on a distributed FileSystem such as HDFS, DBFS, S3, etc. (i.e., `hdfs:///tmp/spark-nlp-assembly-{{ site.sparknlp_version }}.jar`)

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

</div><div class="h3-box" markdown="1">

## Compiled JARs

### Build from source

#### spark-nlp

- FAT-JAR for CPU on Apache Spark 3.0.x, 3.1.x, 3.2.x, 3.3.x, 3.4.x, and 3.5.x

```bash
sbt assembly
```

- FAT-JAR for GPU on Apache Spark 3.0.x, 3.1.x, 3.2.x, 3.3.x, 3.4.x, and 3.5.x

```bash
sbt -Dis_gpu=true assembly
```

- FAT-JAR for M! on Apache Spark 3.0.x, 3.1.x, 3.2.x, 3.3.x, 3.4.x, and 3.5.x

```bash
sbt -Dis_silicon=true assembly
```

</div><div class="h3-box" markdown="1">

### Using the jar manually

If for some reason you need to use the JAR, you can either download the Fat JARs provided here or download it
from [Maven Central](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp).

To add JARs to spark programs use the `--jars` option:

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in
the `spark-packages` section.

## OpenVINO

Spark NLP supports inference and model saving using [OpenVINO](https://docs.openvino.ai/2024/index.html) from version `5.4.2`, enabling optimized inference for specific models.

> OpenVINO is an open-source toolkit for optimizing and deploying deep learning models from cloud to edge. It accelerates deep learning inference across various use cases, such as generative AI, video, audio, and language with models from popular frameworks like PyTorch, TensorFlow, ONNX, and more.

For an example on how to use OpenVINO with Spark NLP, see the [examples folder](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/transformers/openvino).

### Requirements

To run models with OpenVINO, [Intel® Threading Building Blocks (Intel® TBB)](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2021-12/overview.html) needs to be available on your system. If not available, you will run into
"UnsatisfiedLinkError" exceptions during runtime.

For example, to install TBB on Ubuntu we can run

```sh
sudo apt update && sudo apt install libtbb-dev
```

</div>
