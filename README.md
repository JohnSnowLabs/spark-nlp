# Spark NLP: State of the Art Natural Language Processing

[![build](https://github.com/JohnSnowLabs/spark-nlp/workflows/build/badge.svg)](https://github.com/JohnSnowLabs/spark-nlp/actions) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.johnsnowlabs.nlp/spark-nlp_2.11/badge.svg)](https://search.maven.org/artifact/com.johnsnowlabs.nlp/spark-nlp_2.11) [![PyPI version](https://badge.fury.io/py/spark-nlp.svg)](https://badge.fury.io/py/spark-nlp) [![Anaconda-Cloud](https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE)

Spark NLP is a Natural Language Processing library built on top of Apache Spark ML. It provides **simple**, **performant** & **accurate** NLP annotations for machine learning pipelines that **scale** easily in a distributed environment. Spark NLP comes with **1100+** pretrained **pipelines** and **models** in more than **192+** languages. It supports state-of-the-art transformers such as **BERT**, **XLNet**, **ELMO**, **ALBERT**, and **Universal Sentence Encoder** that can be used seamlessly in a cluster. It also offers Tokenization, Word Segmentation, Part-of-Speech Tagging, Named Entity Recognition, Dependency Parsing, Spell Checking, Multi-class Text Classification, Multi-class Sentiment Analysis, Machine Translation (+180 languages), Summarization and Question Answering **(Google T5)**, and many more [NLP tasks](#features).

## Project's website

Take a look at our official Spark NLP page: [http://nlp.johnsnowlabs.com/](http://nlp.johnsnowlabs.com/) for user documentation and examples

## Community support

- [Slack](https://spark-nlp.slack.com/join/shared_invite/zt-j5ttxh0z-Fn3lQSG1Z0KpOs_SRxjdyw#/) For live discussion with the Spark NLP community and the team
- [GitHub](https://github.com/JohnSnowLabs/spark-nlp) Bug reports, feature requests, and contributions
- [Medium](https://medium.com/spark-nlp) Spark NLP articles
- [YouTube](https://www.youtube.com/channel/UCmFOjlpYEhxf_wJUDuz6xxQ/videos) Spark NLP video tutorials

## Table of contents

* [Features](#features)
* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [Apache Spark Support](#apache-spark-support)
* [Databricks Support](#databricks-support)
* [EMR Support](#emr-support)
* [Using Spark NLP](#usage)  
  * [Spark Packages](#spark-packages)
  * [Scala](#scala)
    * [Maven](#maven)
    * [SBT](#sbt)
  * [Python](#python)
    * [Pip/Conda](#pipconda)
  * [Compiled JARs](#compiled-jars)
  * [Apache Zeppelin](#apache-zeppelin)
  * [Jupyter Notebook](#jupyter-notebook-python)
  * [Google Colab Notebook](#google-colab-notebook)
  * [Databricks Cluser](#databricks-cluster)
  * [S3 Cluster](#s3-cluster)  
* [Pipelines & Models](#pipelines-and-models)
  * [Pipelines](#pipelines)
  * [Models](#models)
* [Examples](#examples)  
* [FAQ](#faq)
* [Troubleshooting](#troubleshooting)
* [Acknowledgments](#acknowledgments)
* [Contributing](#contributing)

## Features

* Tokenization
* Trainable Word Segmentation
* Stop Words Removal
* Token Normalizer
* Document Normalizer
* Stemmer
* Lemmatizer
* NGrams
* Regex Matching
* Text Matching
* Chunking
* Date Matcher
* Sentence Detector
* Deep Sentence Detector (Deep learning)
* Dependency parsing (Labeled/unlabeled)
* Part-of-speech tagging
* Sentiment Detection (ML models)
* Spell Checker (ML and DL models)
* Word Embeddings (GloVe and Word2Vec)
* BERT Embeddings (TF Hub models)
* ELMO Embeddings (TF Hub models)
* ALBERT Embeddings (TF Hub models)
* XLNet Embeddings
* Universal Sentence Encoder (TF Hub models)
* BERT Sentence Embeddings (42 TF Hub models)
* Sentence Embeddings
* Chunk Embeddings
* Unsupervised keywords extraction
* Language Detection & Identification (up to 375 languages)
* Multi-class Sentiment analysis (Deep learning)
* Multi-label Sentiment analysis (Deep learning)
* Multi-class Text Classification (Deep learning)
* Neural Machine Translation
* Text-To-Text Transfer Transformer (Google T5)
* Named entity recognition (Deep learning)
* Easy TensorFlow integration
* GPU Support
* Full integration with Spark ML functions
* +710 pre-trained models in +192 languages!
* +450 pre-trained pipelines in +192 languages!
* Multi-lingual NER models: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Hewbrew, Italian, Japanese, Korean, Norwegian, Persian, Polish, Portuguese, Russian, Spanish, Swedish, and Urdu.

## Requirements

In order to use Spark NLP you need the following requirements:

* Java 8
* Apache Spark 2.4.x (or Apache Spark 2.3.x)

## Quick Start

This is a quick example of how to use Spark NLP pre-trained pipeline in Python and PySpark:

```sh
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.6 -y
$ conda activate sparknlp
$ pip install spark-nlp==2.7.0 pyspark==2.4.7
```

In Python console or Jupyter `Python3` kernel:

```python
# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start Spark Session with Spark NLP
# start() functions has two parameters: gpu and spark23
# sparknlp.start(gpu=True) will start the session with GPU support
# sparknlp.start(sparrk23=True) is when you have Apache Spark 2.3.x installed
spark = sparknlp.start()

# Download a pre-trained pipeline
pipeline = PretrainedPipeline('explain_document_dl', lang='en')

# Your testing dataset
text = """
The Mona Lisa is a 16th century oil painting created by Leonardo.
It's held at the Louvre in Paris.
"""

# Annotate your testing dataset
result = pipeline.annotate(text)

# What's in the pipeline
list(result.keys())
Output: ['entities', 'stem', 'checked', 'lemma', 'document',
'pos', 'token', 'ner', 'embeddings', 'sentence']

# Check the results
result['entities']
Output: ['Mona Lisa', 'Leonardo', 'Louvre', 'Paris']
```

For more examples, you can visit our dedicated [repository](https://github.com/JohnSnowLabs/spark-nlp-workshop) to showcase all Spark NLP use cases!

## Apache Spark Support

Spark NLP *2.7.0* has been built on top of Apache Spark 2.4.x and fully supports Apache Spark 2.3.x:

| Spark NLP   |   Apache Spark 2.3.x  | Apache Spark 2.4.x |
|-------------|-----------------------|--------------------|
| 2.6.x       |YES                    |YES                 |
| 2.5.x       |YES                    |YES                 |
| 2.4.x       |Partially              |YES                 |
| 1.8.x       |Partially              |YES                 |
| 1.7.x       |YES                    |NO                  |
| 1.6.x       |YES                    |NO                  |
| 1.5.x       |YES                    |NO                  |

**NOTE:** Starting 2.5.4 release, we support both Apache Spark 2.4.x and Apache Spark 2.3.x at the same time.

Find out more about `Spark NLP` versions from our [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases).

## Databricks Support

Spark NLP 2.7.0 has been tested and is compatible with the following runtimes:

- 6.2
- 6.2 ML
- 6.3
- 6.3 ML
- 6.4
- 6.4 ML
- 6.5
- 6.5 ML

## EMR Support

Spark NLP 2.7.0 has been tested and is compatible with the following EMR releases:

- 5.26.0
- 5.27.0

Full list of [EMR releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-5x.html).

## Usage

## Spark Packages

### Command line (requires internet connection)

This library has been uploaded to the [spark-packages repository](https://spark-packages.org/package/JohnSnowLabs/spark-nlp).

The benefit of spark-packages is that makes it available for both Scala-Java and Python

To use the most recent version on Apache Spark 2.4.x just add the `--packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0` to you spark command:

```sh
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

```sh
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

```sh
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

This can also be used to create a SparkSession manually by using the `spark.jars.packages` option in both Python and Scala.

**NOTE**: To use Spark NLP with GPU you can use the dedicated GPU package `com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.0`

**NOTE**: To use Spark NLP on Apache Spark 2.3.x you should instead use the following packages:

- CPU: `com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.0`
- GPU: `com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:2.7.0`

**NOTE**: In case you are using large pretrained models like UniversalSentenceEncoder, you need to have the following set in your SparkSession:

```sh
spark-shell --driver-memory 16g --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0 spark.kryoserializer.buffer.max 1000M
```

## Scala

Our package is deployed to maven central. To add this package as a dependency in your application:

### Maven

**spark-nlp** on Apache Spark 2.4.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23 -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

### SBT

**spark-nlp** on Apache Spark 2.4.x:

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.7.0"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "2.7.0"
```

**spark-nlp** on Apache Spark 2.3.x:

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-spark23" % "2.7.0"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu-spark23
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu-spark23" % "2.7.0"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

## Python

### Python without explicit Pyspark installation

### Pip/Conda

If you installed pyspark through pip/conda, you can install `spark-nlp` through the same channel.

Pip:

```bash
pip install spark-nlp==2.7.0
```

Conda:

```bash
conda install -c johnsnowlabs spark-nlp
```

PyPI [spark-nlp package](https://pypi.org/project/spark-nlp/) / Anaconda [spark-nlp package](https://anaconda.org/JohnSnowLabs/spark-nlp)

Then you'll have to create a SparkSession either from Spark NLP:

```python
import sparknlp

spark = sparknlp.start()
```

or manually:

```python
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[4]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()
```

If using local jars, you can use `spark.jars` instead for a comma delimited jar files. For cluster setups, of course you'll have to put the jars in a reachable location for all driver and executor nodes.

**Quick example:**

```python
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

#create or get Spark Session

spark = sparknlp.start()

sparknlp.version()
spark.version

#download, load, and annotate a text by pre-trained pipeline

pipeline = PretrainedPipeline('recognize_entities_dl', 'en')
result = pipeline.annotate('The Mona Lisa is a 16th century oil painting created by Leonardo')
```

## Compiled JARs

### Build from source

#### spark-nlp

* FAT-JAR for CPU on Apache Spark 2.4.x

```bash
sbt assembly
```

* FAT-JAR for GPU on Apache Spark 2.4.x

```bash
sbt -Dis_gpu=true assembly
```

* FAT-JAR for CPU on Apache Spark 2.3.x

```bash
sbt -Dis_spark23=true assembly
```

* FAT-JAR for GPU on Apache Spark 2.3.x

```bash
sbt -Dis_gpu=true -Dis_spark23=true assembly
```

### Using the jar manually

If for some reason you need to use the JAR, you can either download the Fat JARs provided here or download it from [Maven Central](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp).

To add JARs to spark programs use the `--jars` option:

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in the `spark-packages` section.

## Apache Zeppelin

Use either one of the following options

- Add the following Maven Coordinates to the interpreter's library list

```bash
com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

- Add path to pre-built jar from [here](#compiled-jars) in the interpreter's library list making sure the jar is available to driver path

### Python in Zeppelin

Apart from previous step, install python module through pip

```bash
pip install spark-nlp==2.7.0
```

Or you can install `spark-nlp` from inside Zeppelin by using Conda:

```bash
python.conda install -c johnsnowlabs spark-nlp
```

Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and install   the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as shown earlier, since it includes both scala and python side installation.

## Jupyter Notebook (Python)

The easiest way to get this done is by making Jupyter Notebook run using pyspark as follows:

```bash
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

Alternatively, you can mix in using `--jars` option for pyspark + `pip install spark-nlp`

If not using pyspark at all, you'll have to run the instructions pointed [here](#python-without-explicit-Pyspark-installation)

## Google Colab Notebook

Google Colab is perhaps the easiest way to get started with spark-nlp. It requires no installation or set up other than having a Google account.

Run the following code in Google Colab notebook and start using spark-nlp right away.

```python
import os

# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! java -version

# Install pyspark
! pip install --ignore-installed pyspark==2.4.7

# Install Spark NLP
! pip install --ignore-installed spark-nlp==2.7.0

# Quick SparkSession start
import sparknlp
spark = sparknlp.start()

print("Spark NLP version")
sparknlp.version()
print("Apache Spark version")
spark.version
```

[Here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs sentiment analysis and NER using pretrained spark-nlp models.

## Databricks Cluster

1. Create a cluster if you don't have one already

2. On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab:

```bash
spark.kryoserializer.buffer.max 1000M
spark.serializer org.apache.spark.serializer.KryoSerializer
```

3. Check `Enable autoscaling local storage` box to have persistent local storage
    
4. In `Libraries` tab inside your cluster you need to follow these steps:

    4.1. Install New -> PyPI -> `spark-nlp` -> Install

    4.2. Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0` -> Install

5. Now you can attach your notebook to the cluster and use Spark NLP!

## S3 Cluster

### With no Hadoop configuration

If your distributed storage is S3 and you don't have a standard Hadoop configuration (i.e. fs.defaultFS)
You need to specify where in the cluster distributed storage you want to store Spark NLP's tmp files.
First, decide where you want to put your *application.conf* file

```scala
import com.johnsnowlabs.util.ConfigLoader
ConfigLoader.setConfigPath("/somewhere/to/put/application.conf")
```

And then we need to put in such application.conf the following content

```bash
sparknlp {
  settings {
    cluster_tmp_dir = "somewhere in s3n:// path to some folder"
  }
}
```

## Pipelines and Models

### Pipelines

Spark NLP offers more than `450+ pre-trained pipelines` in `192 languages`.

**English pipelines:**
| Pipeline                     | Name                                  | Build            | lang |
|:-----------------------------------------|:--------------------------|:-----------------|:------
| Explain Document ML          | `explain_document_ml`                 | 2.4.0 |   `en`    |
| Explain Document DL          | `explain_document_dl`                 | 2.4.3 |   `en`    |
| Recognize Entities DL        | `recognize_entities_dl`               | 2.4.3 |   `en`    |
| Recognize Entities DL        | `recognize_entities_bert`             | 2.4.3 |   `en`    |
| OntoNotes Entities Small     | `onto_recognize_entities_sm`          | 2.4.0 |   `en`    |
| OntoNotes Entities Large     | `onto_recognize_entities_lg`          | 2.4.0 |   `en`    |
| Match Datetime               | `match_datetime`                      | 2.4.0 |   `en`    |
| Match Pattern                | `match_pattern`                       | 2.4.0 |   `en`    |
| Match Chunk                  | `match_chunks`                        | 2.4.0 |   `en`    |
| Match Phrases                | `match_phrases`                       | 2.4.0 |   `en`    |
| Clean Stop                   | `clean_stop`                          | 2.4.0 |   `en`    |
| Clean Pattern                | `clean_pattern`                       | 2.4.0 |   `en`    |
| Clean Slang                  | `clean_slang`                         | 2.4.0 |   `en`    |
| Check Spelling               | `check_spelling`                      | 2.4.0 |   `en`    |
| Check Spelling DL            | `check_spelling_dl`                   | 2.5.0 |   `en`    |
| Analyze Sentiment            | `analyze_sentiment`                   | 2.4.0 |   `en`    |
| Analyze Sentiment DL         | `analyze_sentimentdl_use_imdb`        | 2.5.0 |   `en`    |
| Analyze Sentiment DL         | `analyze_sentimentdl_use_twitter`     | 2.5.0 |   `en`    |
| Dependency Parse             | `dependency_parse`                    | 2.4.0 |   `en`    |

**Quick example:**

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("explain_document_dl", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()
/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.5.0
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_dl,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 10 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|             checked|               lemma|                stem|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Google has announ...|[[document, 0, 10...|[[token, 0, 5, Go...|[[document, 0, 10...|[[token, 0, 5, Go...|[[token, 0, 5, Go...|[[token, 0, 5, go...|[[pos, 0, 5, NNP,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 5, Go...|
|  2|The Paris metro w...|[[document, 0, 11...|[[token, 0, 2, Th...|[[document, 0, 11...|[[token, 0, 2, Th...|[[token, 0, 2, Th...|[[token, 0, 2, th...|[[pos, 0, 2, DT, ...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 4, 8, Pa...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+----------------------------------+
|result                            |
+----------------------------------+
|[Google, TensorFlow]              |
|[Donald John Trump, United States]|
+----------------------------------+
*/
```

#### Please check out our Models Hub for the full list of [pre-trained pipelines](https://nlp.johnsnowlabs.com/models) with examples, demos, benchmarks, and more

### Models

Spark NLP offers more than `710+ pre-trained models` in `192 languages`.

**Some of the selected languages:** `Afrikaans, Arabic, Armenian, Basque, Bengali, Breton, Bulgarian, Catalan, Czech, Dutch, English, Esperanto, Finnish, French, Galician, German, Greek, Hausa, Hebrew, Hindi, Hungarian, Indonesian, Irish, Italian, Japanese, Latin, Latvian, Marathi, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Somali, Southern Sotho, Spanish, Swahili, Swedish, Tswana, Turkish, Ukrainian, Zulu`

**English Models:**

| Model                                    | Name                      | Build            | Lang |
|:-----------------------------------------|:--------------------------|:-----------------|:------
| LemmatizerModel (Lemmatizer)             | `lemma_antbnc`            | 2.0.2 |      `en`
| PerceptronModel (POS)                    | `pos_anc`                 | 2.0.2 |      `en`
| PerceptronModel (POS UD)                 | `pos_ud_ewt`              | 2.2.2 |       `en`
| NerCrfModel (NER with GloVe)             | `ner_crf`                 | 2.4.0 |      `en`
| NerDLModel (NER with GloVe)              | `ner_dl`                  | 2.4.3 |      `en`
| NerDLModel (NER with BERT)               | `ner_dl_bert`             | 2.4.3 |      `en`
| NerDLModel (OntoNotes with GloVe 100d)   | `onto_100`                | 2.4.0 |      `en`
| NerDLModel (OntoNotes with GloVe 300d)   | `onto_300`                | 2.4.0 |      `en`
| SymmetricDeleteModel (Spell Checker)     | `spellcheck_sd`           | 2.0.2 |      `en`
| NorvigSweetingModel (Spell Checker)      | `spellcheck_norvig`       | 2.0.2 |      `en`
| ViveknSentimentModel (Sentiment)         | `sentiment_vivekn`        | 2.0.2 |      `en`
| DependencyParser (Dependency)            | `dependency_conllu`       | 2.0.8 |      `en`
| TypedDependencyParser (Dependency)       | `dependency_typed_conllu` | 2.0.8 |      `en`

**Embeddings:**

| Model    | Name                      | Build            | Lang 
|:--------------|:--------------------------|:-----------------|:------------|
| WordEmbeddings (GloVe)            | `glove_100d`              | 2.4.0 |      `en`  
| BertEmbeddings                    | `bert_base_uncased`       | 2.4.0 |      `en`  
| BertEmbeddings                    | `bert_base_cased`         | 2.4.0 |      `en`  
| BertEmbeddings                    | `bert_large_uncased`      | 2.4.0 |      `en`  
| BertEmbeddings                    | `bert_large_cased`        | 2.4.0 |      `en`  
| ElmoEmbeddings                    | `elmo`                    | 2.4.0 |      `en`  
| UniversalSentenceEncoder  (USE)   | `tfhub_use`              | 2.4.0 |       `en`  
| UniversalSentenceEncoder  (USE)   | `tfhub_use_lg`           | 2.4.0 |       `en`  
| AlbertEmbeddings                  | `albert_base_uncased`    | 2.5.0 |       `en`
| AlbertEmbeddings                  | `albert_large_uncased`    | 2.5.0 |      `en`  
| AlbertEmbeddings                  | `albert_xlarge_uncased`    | 2.5.0 |     `en`
| AlbertEmbeddings                  | `albert_xxlarge_uncased`    | 2.5.0 |    `en`
| XlnetEmbeddings                  | `xlnet_base_cased`    | 2.5.0 |           `en`
| XlnetEmbeddings                  | `xlnet_large_cased`    | 2.5.0 |          `en`

**Classification:**

| Model    | Name                      | Build            | Lang 
|:--------------|:--------------------------|:-----------------|:------------|
| ClassifierDL (with tfhub_use)          | `classifierdl_use_trec6`        | 2.5.0 |      `en`
| ClassifierDL (with tfhub_use)          | `classifierdl_use_trec50`       | 2.5.0 |      `en`
| SentimentDL (with tfhub_use)           | `sentimentdl_use_imdb`          | 2.5.0 |      `en`
| SentimentDL (with tfhub_use)           | `sentimentdl_use_twitter`       | 2.5.0 |      `en`
| SentimentDL (with glove_100d)          | `sentimentdl_glove_imdb`         | 2.5.0 |     `en`

**Quick online example:**

```python
# load NER model trained by deep learning approach and GloVe word embeddings
ner_dl = NerDLModel.pretrained('ner_dl')
# load NER model trained by deep learning approach and BERT word embeddings
ner_bert = NerDLModel.pretrained('ner_dl_bert')
```

```scala
// load French POS tagger model trained by Universal Dependencies
val french_pos = PerceptronModel.pretrained("pos_ud_gsd", lang="fr")
// load Italain LemmatizerModel
val italian_lemma = LemmatizerModel.pretrained("lemma_dxc", lang="it")
````

**Quick offline example:**

* Loading `PerceptronModel` annotator model inside Spark NLP Pipeline

```scala
val french_pos = PerceptronModel.load("/tmp/pos_ud_gsd_fr_2.0.2_2.4_1556531457346/")
      .setInputCols("document", "token")
      .setOutputCol("pos")
```

#### Please check out our Models Hub for the full list of [pre-trained models](https://nlp.johnsnowlabs.com/models) with examples, demo, benchmark, and more

## Examples

Need more **examples**? Check out our dedicated [Spark NLP Showcase](https://github.com/JohnSnowLabs/spark-nlp-workshop) repository to showcase all Spark NLP use cases!

In addition, don't forget to check [Spark NLP in Action](https://demo.johnsnowlabs.com/) built by Streamlit.

### All examples: [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop)

## FAQ

[Check our Articles and FAQ page here](https://nlp.johnsnowlabs.com/articles.html)

## Acknowledgments

### Special community acknowledgments

Thanks in general to the community who have been lately reporting important issues and pull request with bugfixes.
The community has been key in the last releases with feedback in various Spark-based environments.

Here a few specific mentions for recurring feedback and slack participation

* [@maziyarpanahi](https://github.com/maziyarpanahi) - For contributing with testing and valuable feedback
* [@easimadi](https://github.com/easimadi) - For contributing with documentation and valuable feedback

## Contributing

We appreciate any sort of contributions:

* ideas
* feedback
* documentation
* bug reports
* NLP training and testing corpora
* development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

## Contact

nlp@johnsnowlabs.com

## John Snow Labs

[http://johnsnowlabs.com](http://johnsnowlabs.com)
