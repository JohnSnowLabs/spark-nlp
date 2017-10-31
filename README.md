# spark-nlp
John Snow Labs Spark-NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

# Project's website
take a look at our official spark-nlp page: http://nlp.johnsnowlabs.com/ for user documentation and examples

# Usage

## spark-packages

This library has been uploaded to the spark-packages repository https://spark-packages.org/package/JohnSnowLabs/spark-nlp .

To use the most recent version just add the `--packages JohnSnowLabs:spark-nlp:1.0.0` to you spark command

```sh
spark-shell --packages JohnSnowLabs:spark-nlp:1.2.3
```

```sh
pyspark --packages JohnSnowLabs:spark-nlp:1.2.3
```

```sh
spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3
```

Check the package for the published versions if you want to use and old version.

## Bintray repository

If you have a scala project which has a dependency on spark-nlp you can resolve it using our **bintray** repository

Using sbt 0.13.6+

```scala
resolvers += Resolver.bintrayRepo("johnsnowlabs", "johnsnowlabs")
```

For sbt versions before 0.13.6, you need to include the `sbt-bintray` plugin in your project: https://github.com/sbt/sbt-bintray.

After you can add the resolver as before

```scala
resolvers += Resolver.bintrayRepo("johnsnowlabs", "johnsnowlabs")
```

## Using the jar manually 

If for some reason you need to use the jar, you can download the jar from the project's website: http://nlp.johnsnowlabs.com/

From there you can use it in your project setting the `--classpath`

To add jars to spark programs use the `--jars` option

```sh
spark-shell --jars spark-nlp.jar
```

The preferred way to use the library when running spark programs is using the `--packages` option as specified in the `spark-packages` section.

# Contribute
We appreciate any sort of contributions:
* ideas
* feedback
* documentation
* bug reports
* nlp training and testing corpora
* development and testing

Clone the repo and submit your pull-requests! Or directly create issues in this repo.

# Contact
nlp@johnsnowlabs.com

# John Snow Labs
http://johnsnowlabs.com/