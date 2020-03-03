# Spark NLP

[![Build Status](https://travis-ci.org/JohnSnowLabs/spark-nlp.svg?branch=master)](https://travis-ci.org/JohnSnowLabs/spark-nlp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.johnsnowlabs.nlp/spark-nlp_2.11/badge.svg)](https://search.maven.org/artifact/com.johnsnowlabs.nlp/spark-nlp_2.11) [![PyPI version](https://badge.fury.io/py/spark-nlp.svg)](https://badge.fury.io/py/spark-nlp) [![Anaconda-Cloud](https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE)

John Snow Labs Spark NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

## Project's website

Take a look at our official Spark NLP page: [http://nlp.johnsnowlabs.com/](http://nlp.johnsnowlabs.com/) for user documentation and examples

## Slack community channel

[Join Slack](https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLWVjNWUzOGNlODg1Y2FkNGEzNDQ1NDJjMjc3Y2FkOGFmN2Q3ODIyZGVhMzU0NGM3NzRjNDkyZjZlZTQ0YzY1N2I)

## Table of contents

* [Features](#features)
* [Quick Start](#quick-start)
* [Using Spark NLP](#usage)  
  * [Apache Spark Support](#apache-spark-support)
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
* Stop Words Removal
* Normalizer
* Stemmer
* Lemmatizer
* NGrams
* Regex Matching
* Text Matching
* Chunking
* Date Matcher
* Sentence Detector
* Part-of-speech tagging
* Sentiment Detection (ML models)
* Spell Checker (ML and DL models)
* Word Embeddings (GloVe and Word2Vec)
* BERT Embeddings (TF Hub models)
* ELMO Embeddings (TF Hub models)
* Universal Sentence Encoder (TF Hub models)
* Sentence Embeddings
* Chunk Embeddings
* Named entity recognition (Deep learning)
* Dependency parsing (Labeled/unlabled)
* Easy TensorFlow integration
* Full integration with Spark ML functions
* +30 pre-trained models in 5 languages (English, French, German, Italian, and Spanish)
* +30 pre-trained pipelines!

## Quick Start

This is a quick example of how to use Spark NLP pre-trained pipeline:

```python
# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start Spark Session with Spark NLP
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

For more examples you can visit our dedicated [repository](https://github.com/JohnSnowLabs/spark-nlp-workshop) to showcase all Spark NLP use cases!

## Usage

## Apache Spark Support

Spark NLP *2.4.1* has been built on top of Apache Spark 2.4.4

| Spark NLP   |   Apache Spark 2.3.x  | Apache Spark 2.4.x |
|-------------|-----------------------|--------------------|
| 2.4.x       |YES**                  |YES                 |
| 1.8.x       |Partially              |YES                 |
| 1.7.x       |YES                    |NO                  |
| 1.6.x       |YES                    |NO                  |
| 1.5.x       |YES                    |NO                  |

Find out more about `Spark NLP` versions from our [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases).

** Spark NLP is built and released based on `Apache Spark 2.4.x`, in order to use it with Apache Spark `2.3.x` you need to manually compile it by changing the version in our `build.sbt` file.

** We do have the Fat JAR of Spark NLP 2.4.0 release already compiled for `Apache Spark 2.3.x` and it can be downloaded from our S3 [from here](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp_2.11-2.4.0_spark-2.3.0.jar).

## Spark Packages

### Command line (requires internet connection)

This library has been uploaded to the [spark-packages repository](https://spark-packages.org/package/JohnSnowLabs/spark-nlp).

Benefit of spark-packages is that makes it available for both Scala-Java and Python

To use the most recent version just add the `--packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1` to you spark command

```sh
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
```

```sh
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
```

```sh
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
```

This can also be used to create a SparkSession manually by using the `spark.jars.packages` option in both Python and Scala.

**NOTE**: To ues SPark NLP with GPU you can ues dedicated package for GPU `com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.4.1`

## Scala

Our package is deployed to maven central. In order to add this package as a dependency in your application:

### Maven

**spark-nlp:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<!-- https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu -->
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.4.1</version>
</dependency>
```

### SBT

**spark-nlp:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.4.1"
```

**spark-nlp-gpu:**

```sbtshell
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "2.4.1"
```

Maven Central: [https://mvnrepository.com/artifact/com.johnsnowlabs.nlp](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp)

## Python

### Python without explicit Pyspark installation

### Pip/Conda

If you installed pyspark through pip/conda, you can install `spark-nlp` through the same channel.

Pip:

```bash
pip install spark-nlp==2.4.1
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
    .appName("ner")\
    .master("local[4]")\
    .config("spark.driver.memory","8G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1")\
    .config("spark.kryoserializer.buffer.max", "500m")\
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
result = pipeline.annotate('Harry Potter is a great movie')
```

## Compiled JARs

### Build from source

#### spark-nlp

* FAT-JAR for CPU

```bash
sbt assembly
```

* FAT-JAR for GPU

```bash
sbt -Dis_gpu=true assembly
```

* Packaging the project

```bash
sbt package
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

* Add the following Maven Coordinates to the interpreter's library list

```bash
com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
```

* Add path to pre-built jar from [here](#compiled-jars) in the interpreter's library list making sure the jar is available to driver path

### Python in Zeppelin

Apart from previous step, install python module through pip

```bash
pip install spark-nlp==2.4.1
```

Or you can install `spark-nlp` from inside Zeppelin by using Conda:

```bash
python.conda install -c johnsnowlabs spark-nlp
```

Configure Zeppelin properly, use cells with %spark.pyspark or any interpreter name you chose.

Finally, in Zeppelin interpreter settings, make sure you set properly zeppelin.python to the python you want to use and installed the pip library with (e.g. `python3`).

An alternative option would be to set `SPARK_SUBMIT_OPTIONS` (zeppelin-env.sh) and make sure `--packages` is there as shown earlier, since it includes both scala and python side installation.

## Jupyter Notebook (Python)

Easiest way to get this done is by making Jupyter Notebook run using pyspark as follows:

```bash
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook

pyspark --packages JohnSnowLabs:spark-nlp:2.4.1
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
! pip install --ignore-installed pyspark==2.4.4

# Install Spark NLP
! pip install --ignore-installed spark-nlp==2.4.1

# Quick SparkSession start
import sparknlp
spark = sparknlp.start()

print("Spark NLP version")
sparknlp.version()
print("Apache Spark version")
spark.version
```

[Here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) is a live demo on Google Colab that performs sentiment analysis and NER using pretrained spark-nlp models.

## S3 Cluster

### With no hadoop configuration

If your distributed storage is S3 and you don't have a standard hadoop configuration (i.e. fs.defaultFS)
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

Spark NLP offers more than `30 pre-trained pipelines` in `5 languages`.

**English pipelines:**

| Pipelines            | Name                   |
| -------------------- | ---------------------- |
| Explain Document ML  | `explain_document_ml`  |
| Explain Document DL | `explain_document_dl`  |
| Explain Document DL Fast | `explain_document_dl_fast`  |
| Recognize Entities DL | `recognize_entities_dl` |
| OntoNotes Entities Small | `onto_recognize_entities_sm` |
| OntoNotes Entities Large | `onto_recognize_entities_lg` |
| Match Datetime | `match_datetime` |
| Match Pattern | `match_pattern` |
| Match Chunk | `match_chunks` |
| Match Phrases | `match_phrases`|
| Clean Stop | `clean_stop`|
| Clean Pattern | `clean_pattern`|
| Clean Slang | `clean_slang`|
| Check Spelling | `check_spelling`|
| Analyze Sentiment | `analyze_sentiment` |
| Dependency Parse | `dependency_parse` |

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
2.0.8
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

#### Please check our dedicated repo for a full list of [pre-trained pipelines](https://github.com/JohnSnowLabs/spark-nlp-models)

### Models

Spark NLP offers more than `30 pre-trained models` in `5 languages`.

**English pipelines:**

| Model                                  |   Name     |
|----------------------------------------|------------|
|LemmatizerModel (Lemmatizer)            |  `lemma_antbnc`      |
|PerceptronModel (POS)                   |   `pos_anc`     |
|NerCRFModel (NER with GloVe)            |    `ner_crf`    |
|NerDLModel (NER with GloVe)             |    `ner_dl`    |
|NerDLModel (NER with BERT)| `ner_dl_bert_base_cased`|
|NerDLModel (OntoNotes with GloVe 100d)| `onto_100`|
|NerDLModel (OntoNotes with GloVe 300d)| `onto_300`|
|WordEmbeddings (GloVe) | `glove_100d` |
|BertEmbeddings (base_uncased) | `bert_base_uncased` |
|BertEmbeddings (base_cased) | `bert_base_cased` |
|BertEmbeddings (large_uncased) | `bert_large_uncased` |
|BertEmbeddings (large_cased) | `bert_large_cased` |
|DeepSentenceDetector| `ner_dl_sentence`|
|ContextSpellCheckerModel (Spell Checker)|   `spellcheck_dl`     |
|SymmetricDeleteModel (Spell Checker)    |   `spellcheck_sd`     |
|NorvigSweetingModel (Spell Checker)     |  `spellcheck_norvig`   |
|ViveknSentimentModel (Sentiment)        |    `sentiment_vivekn`    |
|DependencyParser (Dependency)        |    `dependency_conllu`    |
|TypedDependencyParser (Dependency)        |    `dependency_typed_conllu`    |

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

#### Please check our dedicated repo for a full list of [pre-trained models](https://github.com/JohnSnowLabs/spark-nlp-models)

## Examples

Need more **examples**? Check out our dedicated [repository](https://github.com/JohnSnowLabs/spark-nlp-workshop) to showcase all Spark NLP use cases!

### All examples: [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop)

## FAQ

[Check our Articles and FAQ page here](https://nlp.johnsnowlabs.com/articles.html)

## Troubleshooting

### ```TypeError: 'JavaPackage' object is not callable```

If you get this common python error, it means that the Spark NLP was not loaded correctly in your session, take a look at the following suggestions for a solution
(Thanks Eric Nelson for putting this together)

1. Make sure you are using Apache Spark 2.4.4 (or whatever version Spark NLP was compiled with)
2. Make sure your SPARK_HOME and PATH environment variables are pointing to such Spark and not any other installation on your system
3. If on Windows, download Hadoop winutils.exe and add it to your PATH: https://github.com/steveloughran/winutils
4. HADOOP_HOME should also be set in some cases, pointing to your SPARK_HOME should work if you don't have an explicit hadoop installation
5. If you are running `pyspark` instead of just `jupyter notebook`, make sure you setup `PYSPARK_DRIVER_PYTHON`, `PYSPARK_DRIVER_PYTHON_OPTS` and `PYSPARK_PYTHON` as pointed in the documentation
6. `pip install spark-nlp==2.4.1` even if you are using `--packages` as a safety instruction
7. Make sure all dependencies are properly written and/or paths to any jars you are manually providing. Spark does not fail upon wrong path, it will just ignore it
8. If you get dependency failures when starting Spark, make sure to add antivirus and firewall exceptions. Windows antivirus adversely impacts performance when resolving dependencies.


## Acknowledgments

### Special community acknowledgments

Thanks in general to the community who have been lately reporting important issues and pull request with bugfixes.
Community has been key in the last releases with feedback in various Spark based environments.

Here a few specific mentions for recurring feedback and slack participation

* [@maziyarpanahi](https://github.com/maziyarpanahi) - For contributing with testing and valuable feedback
* [@easimadi](https://github.com/easimadi) - For contributing with documentation and valuable feedback

## Contributing

We appreciate any sort of contributions:

* ideas
* feedback
* documentation
* bug reports
* nlp training and testing corpora
* development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

## Contact

nlp@johnsnowlabs.com

## John Snow Labs

[http://johnsnowlabs.com](http://johnsnowlabs.com)
