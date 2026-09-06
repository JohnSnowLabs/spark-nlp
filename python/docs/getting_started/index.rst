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
    # Apache Spark 3.x (Scala 2.12)
    spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|
    # Apache Spark 4.x (Scala 2.13)
    spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.13:|release|

    # Load Spark NLP with PySpark
    # Apache Spark 3.x (Scala 2.12)
    pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|
    # Apache Spark 4.x (Scala 2.13)
    pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.13:|release|

    # Load Spark NLP with Spark Submit
    # Apache Spark 3.x (Scala 2.12)
    spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:|release|
    # Apache Spark 4.x (Scala 2.13)
    spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.13:|release|

    # Spark 4.0.0 / Scala 2.13
    spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-spark400_2.13:|release|

    # Spark 4.0.1 and later validated Spark 4 versions / Scala 2.13
    spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.13:|release|

    # Load Spark NLP as external JAR after compiling and building Spark NLP by `sbt assembly`
    spark-shell --jar spark-nlp-assembly-|release|


************
Requirements
************

Spark NLP supports explicit Spark/Scala runtime lanes:

* Spark 3.x with Scala 2.12
* Spark 4.0.0 with Scala 2.13 through the dedicated ``spark400`` artifact
* Spark 4.0.1, 4.1.0, 4.1.1, and 4.1.2 with Scala 2.13 through the default ``_2.13`` artifact

Spark 3.x with Scala 2.13 and Spark 4.x with Scala 2.12 are not supported. Use a
Java and Python version supported by the selected Apache Spark runtime; Spark 4
validation uses Java 17.

It is recommended to have basic knowledge of the framework and a working environment before using Spark NLP.
Please refer to `Spark documentation <https://spark.apache.org/docs/latest/api/python/index.html>`_ to get started with Spark.

************
Installation
************

First, make sure the installed Java version matches the selected Spark runtime.
Use the existing Java 8/11 support baseline for Spark 3.x and Java 17 for Spark 4.x:

.. code-block:: bash

    java -version
    # Spark 3.x: Java 8/11 baseline
    # Spark 4.x: Java 17

Using Conda
===========

Let’s create a new `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment to manage all the dependencies there.

Then we can create a new environment ``sparknlp`` and install the ``spark-nlp`` package with pip:

.. code-block:: bash
    :substitutions:

    conda create -n sparknlp python=3.9 -y
    conda activate sparknlp
    # Apache Spark 3.x (Scala 2.12)
    conda install -c johnsnowlabs spark-nlp==|release| pyspark==|pyspark3_version| jupyter
    # Apache Spark 4.x (Scala 2.13)
    conda install -c johnsnowlabs spark-nlp==|release| pyspark==|pyspark4_version| jupyter

Now you should be ready to create a jupyter notebook with Spark NLP running:

.. code-block:: bash

    jupyter notebook

Using Virtualenv
================

We can also create a Python `Virtualenv <https://virtualenv.pypa.io/en/latest/>`_:

.. code-block:: bash
    :substitutions:

    virtualenv sparknlp --python=python3.9 # depends on how your Python installation is set up
    source sparknlp/bin/activate
    # Apache Spark 3.x (Scala 2.12)
    pip install spark-nlp==|release| pyspark==|pyspark3_version| jupyter
    # Apache Spark 4.x (Scala 2.13)
    pip install spark-nlp==|release| pyspark==|pyspark4_version| jupyter

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

The same Python wheel supports the declared Spark 3 and Spark 4 lanes.
``sparknlp.start()`` detects the installed PySpark version and resolves:

* Spark 3.x to ``spark-nlp_2.12``
* Spark 4.0.0 to ``spark-nlp-spark400_2.13``
* Spark 4.0.1, 4.1.0, 4.1.1, and 4.1.2 to ``spark-nlp_2.13``

The ``gpu``, ``apple_silicon``, and ``aarch64`` options select the corresponding
hardware artifact in the same runtime lane. Only one hardware option may be enabled.

The release line that introduces this mapping removes the experimental
``scala213`` argument. Install a supported PySpark version and allow
``sparknlp.start()`` to select Scala and the Maven lane automatically.

If you need to manually start SparkSession because you have other configurations and ``sparknlp.start()`` is not including them,
you can manually start the SparkSession with:

.. code-block:: python
    :substitutions:

    SparkSession.builder \
        .appName("Spark NLP") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.13:|release|") \
        .getOrCreate()

The manual example above is for Spark 3.x. Use
``spark-nlp-spark400_2.13`` for Spark 4.0.0 or ``spark-nlp_2.13`` for
Spark 4.0.1, 4.1.0, 4.1.1, or 4.1.2.
