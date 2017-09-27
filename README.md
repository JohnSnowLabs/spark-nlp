# spark-nlp
John Snow Labs Spark-NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

# website
take a look at our official spark-nlp page: http://nlp.johnsnowlabs.com/ for user documentation and examples

# Use
as of now, the only way to use the library is either building it yourself or by including the downloadable jar in spark classpath, which can be downloaded here: https://github.com/JohnSnowLabs/spark-nlp/raw/master/docs/releases/spark-nlp-snapshot.jar

run spark-shell or spark-submit with **--jars /path/to/spark-nlp-snapshot.jar** to use the library in scala

we are working on publishing the library on public repos to make it easier to use and enable
spark packages. This will allow to use pyspark library as well

To use pyspark now, you may have to clone the repo, and stand inside the python folder to make sparknlp module avaiable, while also adding the jar to pyspark with --jars as above

Stay tuned for this section to be updated


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