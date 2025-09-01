# Spark NLP: State-of-the-Art Natural Language Processing & LLMs Library

<p align="center">
    <a href="https://github.com/JohnSnowLabs/spark-nlp/actions" alt="build">
        <img src="https://github.com/JohnSnowLabs/spark-nlp/workflows/build/badge.svg" /></a>
    <a href="https://github.com/JohnSnowLabs/spark-nlp/releases" alt="Current Release Version">
        <img src="https://img.shields.io/github/v/release/JohnSnowLabs/spark-nlp.svg?style=flat-square&logo=github" /></a>
    <a href="https://search.maven.org/artifact/com.johnsnowlabs.nlp/spark-nlp_2.12" alt="Maven Central">
        <img src="https://maven-badges.herokuapp.com/maven-central/com.johnsnowlabs.nlp/spark-nlp_2.12/badge.svg" /></a>
    <a href="https://badge.fury.io/py/spark-nlp" alt="PyPI version">
        <img src="https://badge.fury.io/py/spark-nlp.svg" /></a>
    <a href="https://anaconda.org/JohnSnowLabs/spark-nlp" alt="Anaconda-Cloud">
        <img src="https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg" /></a>
    <a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
    <a href="https://pypi.org/project/spark-nlp/" alt="PyPi downloads">
        <img src="https://static.pepy.tech/personalized-badge/spark-nlp?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads" /></a>
</p>

Spark NLP is a state-of-the-art Natural Language Processing library built on top of Apache Spark. It provides **simple**, **performant** & **accurate** NLP annotations for machine learning pipelines that **scale** easily in a distributed environment.

Spark NLP comes with **100000+** pretrained **pipelines** and **models** in more than **200+** languages.
It also offers tasks such as **Tokenization**, **Word Segmentation**, **Part-of-Speech Tagging**, Word and Sentence **Embeddings**, **Named Entity Recognition**, **Dependency Parsing**, **Spell Checking**, **Text Classification**, **Sentiment Analysis**, **Token Classification**, **Machine Translation** (+180 languages), **Summarization**, **Question Answering**, **Table Question Answering**, **Text Generation**, **Image Classification**, **Image to Text (captioning)**, **Automatic Speech Recognition**, **Zero-Shot Learning**, and many more [NLP tasks](#features).

**Spark NLP** is the only open-source NLP library in **production** that offers state-of-the-art transformers such as **BERT**, **CamemBERT**, **ALBERT**, **ELECTRA**, **XLNet**, **DistilBERT**, **RoBERTa**, **DeBERTa**, **XLM-RoBERTa**, **Longformer**, **ELMO**, **Universal Sentence Encoder**, **Llama-2**, **M2M100**, **BART**, **Instructor**, **E5**, **Google T5**, **MarianMT**, **OpenAI GPT2**, **Vision Transformers (ViT)**, **OpenAI Whisper**, **Llama**, **Mistral**, **Phi**, **Qwen2**, and many more not only to **Python** and **R**, but also to **JVM** ecosystem (**Java**, **Scala**, and **Kotlin**) at **scale** by extending **Apache Spark** natively.

## Model Importing Support

Spark NLP provides easy support for importing models from various popular frameworks:

- **TensorFlow**
- **ONNX**
- **OpenVINO**
- **Llama.cpp (GGUF)**

This wide range of support allows you to seamlessly integrate models from different sources into your Spark NLP workflows, enhancing flexibility and compatibility with existing machine learning ecosystems.

## Project's website

Take a look at our official Spark NLP page: [https://sparknlp.org/](https://sparknlp.org/) for user
documentation and examples

## Features

- [Text Preprocessing](https://sparknlp.org/docs/en/features#text-preproccesing)
- [Parsing and Analysis](https://sparknlp.org/docs/en/features#parsing-and-analysis)
- [Sentiment and Classification](https://sparknlp.org/docs/en/features#sentiment-and-classification)
- [Embeddings](https://sparknlp.org/docs/en/features#embeddings)
- [Classification and Question Answering](https://sparknlp.org/docs/en/features#classification-and-question-answering-models)
- [Machine Translation and Generation](https://sparknlp.org/docs/en/features#machine-translation-and-generation)
- [Image and Speech](https://sparknlp.org/docs/en/features#image-and-speech)
- [Integration and Interoperability (ONNX, OpenVINO)](https://sparknlp.org/docs/en/features#integration-and-interoperability)
- [Pre-trained Models (36000+ in +200 languages)](https://sparknlp.org/docs/en/features#pre-trained-models)
- [Multi-lingual Support](https://sparknlp.org/docs/en/features#multi-lingual-support)

## Quick Start

This is a quick example of how to use a Spark NLP pre-trained pipeline in Python and PySpark:

```sh
$ java -version
# should be Java 8 or 11 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.7 -y
$ conda activate sparknlp
# spark-nlp by default is based on pyspark 3.x
$ pip install spark-nlp==6.1.3 pyspark==3.3.1
```

In Python console or Jupyter `Python3` kernel:

```python
# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start SparkSession with Spark NLP
# start() functions has 3 parameters: gpu, apple_silicon, and memory
# sparknlp.start(gpu=True) will start the session with GPU support
# sparknlp.start(apple_silicon=True) will start the session with macOS M1 & M2 support
# sparknlp.start(memory="16G") to change the default driver memory in SparkSession
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

For more examples, you can visit our dedicated [examples](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples) to showcase all Spark NLP use cases!

### Packages Cheatsheet

This is a cheatsheet for corresponding Spark NLP Maven package to Apache Spark / PySpark major version:

| Apache Spark            | Spark NLP on CPU   | Spark NLP on GPU           | Spark NLP on AArch64 (linux)   | Spark NLP on Apple Silicon           |
|-------------------------|--------------------|----------------------------|--------------------------------|--------------------------------------|
| 3.0/3.1/3.2/3.3/3.4/3.5 | `spark-nlp`        | `spark-nlp-gpu`            | `spark-nlp-aarch64`            | `spark-nlp-silicon`                  |
| Start Function          | `sparknlp.start()` | `sparknlp.start(gpu=True)` | `sparknlp.start(aarch64=True)` | `sparknlp.start(apple_silicon=True)` |

NOTE: `M1/M2` and `AArch64` are under `experimental` support. Access and support to these architectures are limited by the
community and we had to build most of the dependencies by ourselves to make them compatible. We support these two
architectures, however, they may not work in some environments.

## Pipelines and Models

For a quick example of using pipelines and models take a look at our official [documentation](https://sparknlp.org/docs/en/install#pipelines-and-models)

#### Please check out our Models Hub for the full list of [pre-trained models](https://sparknlp.org/models) with examples, demo, benchmark, and more

## Platform and Ecosystem Support

### Apache Spark Support

Spark NLP *6.1.3* has been built on top of Apache Spark 3.4 while fully supports Apache Spark 3.0.x, 3.1.x, 3.2.x, 3.3.x, 3.4.x, and 3.5.x

| Spark NLP | Apache Spark 3.5.x | Apache Spark 3.4.x | Apache Spark 3.3.x | Apache Spark 3.2.x | Apache Spark 3.1.x | Apache Spark 3.0.x | Apache Spark 2.4.x | Apache Spark 2.3.x |
|-----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 6.x.x and up     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.5.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.4.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.3.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.2.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.1.x     | Partially          | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |
| 5.0.x     | YES                | YES                | YES                | YES                | YES                | YES                | NO                 | NO                 |

Find out more about `Spark NLP` versions from our [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases).

### Scala and Python Support

| Spark NLP | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10| Scala 2.11 | Scala 2.12 |
|-----------|------------|------------|------------|------------|------------|------------|------------|
| 6.0.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.5.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.4.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.3.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.2.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.1.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |
| 5.0.x     | NO         | YES        | YES        | YES        | YES        | NO         | YES        |

Find out more about 4.x `SparkNLP` versions in our official [documentation](https://sparknlp.org/docs/en/install#apache-spark-support)

### Databricks Support

Spark NLP 6.1.3 has been tested and is compatible with the following runtimes:

| **CPU**            | **GPU**            |
|--------------------|--------------------|
| 14.1 / 14.1 ML     | 14.1 ML & GPU      |
| 14.2 / 14.2 ML     | 14.2 ML & GPU      |
| 14.3 / 14.3 ML     | 14.3 ML & GPU      |
| 15.0 / 15.0 ML     | 15.0 ML & GPU      |
| 15.1 / 15.1 ML     | 15.1 ML & GPU      |
| 15.2 / 15.2 ML     | 15.2 ML & GPU      |
| 15.3 / 15.3 ML     | 15.3 ML & GPU      |
| 15.4 / 15.4 ML     | 15.4 ML & GPU      |
| 16.4 / 16.4 ML     | 16.4 ML & GPU      |

We are compatible with older runtimes. For a full list check databricks support in our official [documentation](https://sparknlp.org/docs/en/install#databricks-support)

### EMR Support

Spark NLP 6.1.3 has been tested and is compatible with the following EMR releases:

| **EMR Release**    |
|--------------------|
| emr-6.13.0         |
| emr-6.14.0         |
| emr-6.15.0         |
| emr-7.0.0          |
| emr-7.1.0          |
| emr-7.2.0          |
| emr-7.3.0          |
| emr-7.4.0          |
| emr-7.5.0          |
| emr-7.6.0          |
| emr-7.7.0          |
| emr-7.8.0          |


We are compatible with older EMR releases. For a full list check EMR support in our official [documentation](https://sparknlp.org/docs/en/install#emr-support)

Full list of [Amazon EMR 6.x releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-6x.html)
Full list of [Amazon EMR 7.x releases](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-release-7x.html)

NOTE: The EMR 6.1.0 and 6.1.1 are not supported.

## Installation

### Command line (requires internet connection)

To install spark-nlp packages through command line follow [these instructions](https://sparknlp.org/docs/en/install#command-line) from our official documentation

### Scala

Spark NLP supports Scala 2.12.15 if you are using Apache Spark 3.0.x, 3.1.x, 3.2.x, 3.3.x, and 3.4.x versions. Our packages are
deployed to Maven central. To add any of our packages as a dependency in your application you can follow [these instructions](https://sparknlp.org/docs/en/install#scala-and-java)
from our official documentation.

If you are interested, there is a simple SBT project for Spark NLP to guide you on how to use it in your
projects [Spark NLP Starter](https://github.com/maziyarpanahi/spark-nlp-starter)

### Python

Spark NLP supports Python 3.7.x and above depending on your major PySpark version.
Check all available installations for Python in our official [documentation](https://sparknlp.org/docs/en/install#python)

### Compiled JARs

To compile the jars from source follow [these instructions](https://sparknlp.org/docs/en/compiled#jars) from our official documentation

## Platform-Specific Instructions

For detailed instructions on how to use Spark NLP on supported platforms, please refer to our official documentation:

| Platform                | Supported Language(s) |
|-------------------------|-----------------------|
| [Apache Zeppelin](https://sparknlp.org/docs/en/install#apache-zeppelin)      | Scala, Python         |
| [Jupyter Notebook](https://sparknlp.org/docs/en/install#jupter-notebook) | Python                |
| [Google Colab Notebook](https://sparknlp.org/docs/en/install#google-colab-notebook) | Python                |
| [Kaggle Kernel](https://sparknlp.org/docs/en/install#kaggle-kernel)        | Python                |
| [Databricks Cluster](https://sparknlp.org/docs/en/install#databricks-cluster)    | Scala, Python         |
| [EMR Cluster](https://sparknlp.org/docs/en/install#emr-cluster)           | Scala, Python         |
| [GCP Dataproc Cluster](https://sparknlp.org/docs/en/install#gcp-dataproc) | Scala, Python         |

### Offline

Spark NLP library and all the pre-trained models/pipelines can be used entirely offline with no access to the Internet.
Please check [these instructions](https://sparknlp.org/docs/en/install#s3-integration) from our official documentation
to use Spark NLP offline.

## Advanced Settings

You can change Spark NLP configurations via Spark properties configuration.
Please check [these instructions](https://sparknlp.org/docs/en/install#sparknlp-properties) from our official documentation.

### S3 Integration

In Spark NLP we can define S3 locations to:

- Export log files of training models
- Store tensorflow graphs used in `NerDLApproach`

Please check [these instructions](https://sparknlp.org/docs/en/install#s3-integration) from our official documentation.

## Documentation

### Examples

Need more **examples**? Check out our dedicated [Spark NLP Examples](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples)
repository to showcase all Spark NLP use cases!

Also, don't forget to check [Spark NLP in Action](https://sparknlp.org/demo) built by Streamlit.

#### All examples: [spark-nlp/examples](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples)

### FAQ

[Check our Articles and Videos page here](https://sparknlp.org/learn)

### Citation

We have published a [paper](https://www.sciencedirect.com/science/article/pii/S2665963821000063) that you can cite for
the Spark NLP library:

```bibtex
@article{KOCAMAN2021100058,
    title = {Spark NLP: Natural language understanding at scale},
    journal = {Software Impacts},
    pages = {100058},
    year = {2021},
    issn = {2665-9638},
    doi = {https://doi.org/10.1016/j.simpa.2021.100058},
    url = {https://www.sciencedirect.com/science/article/pii/S2665963.2.300063},
    author = {Veysel Kocaman and David Talby},
    keywords = {Spark, Natural language processing, Deep learning, Tensorflow, Cluster},
    abstract = {Spark NLP is a Natural Language Processing (NLP) library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines that can scale easily in a distributed environment. Spark NLP comes with 1100+ pretrained pipelines and models in more than 192+ languages. It supports nearly all the NLP tasks and modules that can be used seamlessly in a cluster. Downloaded more than 2.7 million times and experiencing 9x growth since January 2020, Spark NLP is used by 54% of healthcare organizations as the world’s most widely used NLP library in the enterprise.}
    }
}
```

## Community support

- [Slack](https://join.slack.com/t/spark-nlp/shared_invite/zt-198dipu77-L3UWNe_AJ8xqDk0ivmih5Q) For live discussion with the Spark NLP community and the team
- [GitHub](https://github.com/JohnSnowLabs/spark-nlp) Bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas,
  and show off how you use Spark NLP!
- [Medium](https://medium.com/spark-nlp) Spark NLP articles
- [YouTube](https://www.youtube.com/channel/UCmFOjlpYEhxf_wJUDuz6xxQ/videos) Spark NLP video tutorials

## Contributing

We appreciate any sort of contributions:

- ideas
- feedback
- documentation
- bug reports
- NLP training and testing corpora
- Development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

## John Snow Labs

[http://johnsnowlabs.com](http://johnsnowlabs.com)
