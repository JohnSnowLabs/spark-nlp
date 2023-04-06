..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

###############
Getting Started
###############

*********************
Spark NLP Cheat Sheet
*********************

This cheat sheet can be used as a quick reference on how to set up your environment:

.. code-block:: bash
    :substitutions:

    # Install Spark NLP from PyPI
    pip install spark-nlp==|release|

    # Install Spark NLP from Anaconda/Conda
    conda install -c johnsnowlabs spark-nlp==|release|

    # Load Spark NLP with Spark Shell
    spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|

    # Load Spark NLP with PySpark
    pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|

    # Load Spark NLP with Spark Submit
    spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|

    # Load Spark NLP as external JAR after compiling and building Spark NLP by `sbt assembly`
    spark-shell --jar spark-nlp-assembly-|release|


************
Requirements
************

Spark NLP is built on top of Apache Spark `3.x`. For using Spark NLP you need:

* Java 8
* Apache Spark (from ``2.3.x`` to ``3.3.x``)
* Python ``3.8.x`` if you are using PySpark ``3.x``

    * **NOTE**: Since Spark version 3.2, Python 3.6 is deprecated. If you are using this
      python version, consider sticking to lower versions of Spark.
    * For Python ``3.6.x`` and ``3.7.x`` we recommend PySpark ``2.3.x`` or ``2.4.x``

It is recommended to have basic knowledge of the framework and a working environment before using Spark NLP.
Please refer to `Spark documentation <https://spark.apache.org/docs/latest/api/python/index.html>`_ to get started with Spark.

************
Installation
************

First, let's make sure the installed java version is Java 8 (Oracle or OpenJDK):

.. code-block:: bash

    java -version
    # openjdk version "1.8.0_292"

Using Conda
===========

Let’s create a new `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment to manage all the dependencies there.

Then we can create a new environment ``sparknlp`` and install the ``spark-nlp`` package with pip:

.. code-block:: bash
    :substitutions:

    conda create -n sparknlp python=3.8 -y
    conda activate sparknlp
    conda install -c johnsnowlabs spark-nlp==|release| pyspark==|pyspark_version| jupyter

Now you should be ready to create a jupyter notebook with Spark NLP running:

.. code-block:: bash

    jupyter notebook

Using Virtualenv
================

We can also create a Python `Virtualenv <https://virtualenv.pypa.io/en/latest/>`_:

.. code-block:: bash
    :substitutions:

    virtualenv sparknlp --python=python3.8 # depends on how your Python installation is set up
    source sparknlp/bin/activate
    pip install spark-nlp==|release| pyspark==|pyspark_version| jupyter

Now you should be ready to create a jupyter notebook with Spark NLP running:

.. code-block:: bash

    jupyter notebook

****************************************
Starting a Spark NLP Session from Python
****************************************

A Spark session for Spark NLP can be created (or retrieved) by using :func:`sparknlp.start`:

.. code-block:: python

    import sparknlp
    spark = sparknlp.start()

If you need to manually start SparkSession because you have other configurations and ``sparknlp.start()`` is not including them,
you can manually start the SparkSession with:

.. code-block:: python
    :substitutions:

    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[4]")\
        .config("spark.driver.memory","16G")\
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.kryoserializer.buffer.max", "2000M")\
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:|release|")\
        .getOrCreate()
