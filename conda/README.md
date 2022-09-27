# spark-nlp-conda

[![Version Status](https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![Last Update](https://anaconda.org/johnsnowlabs/spark-nlp/badges/latest_release_date.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![Downloads](https://pepy.tech/badge/spark-nlp/month)](https://pepy.tech/project/spark-nlp/month) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE)

John Snow Labs Spark NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

## Setup

Before you start, install anaconda-client and conda-build:

```bash
conda install anaconda-client conda-build
```

Make sure you are logged in as JohnSnowLabs

```bash
conda login
```

## Build

Purge the previous builds:

```bash
conda build purge
```

Turn off auto-upload:

```bash
conda config --set anaconda_upload no
```

Build `spark-nlp` from the latest PyPI tar:

```bash
conda build . --python=3.7 && conda build . --python=3.8
```

Example of uploading Conda package to Anaconda Cloud:

```bash
anaconda upload /anaconda3/conda-bld/noarch/spark-nlp-version-py37_0.tar.bz2
```

## Install

Install spark-nlp by using conda:

```bash
conda install -c johnsnowlabs spark-nlp
```

## Main repository

[https://github.com/JohnSnowLabs/spark-nlp](https://github.com/JohnSnowLabs/spark-nlp)

## Project's website

Take a look at our official spark-nlp page: [http://nlp.johnsnowlabs.com/](http://nlp.johnsnowlabs.com/) for user documentation and examples

## License

Apache Licence 2.0
