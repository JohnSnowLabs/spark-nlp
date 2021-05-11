#!/bin/bash

#default values for pyspark, spark-nlp, and SPARK_HOME
SPARKNLP="3.0.3"
PYSPARK="3.0.2"
SPARKHOME="/content/spark-3.1.1-bin-hadoop2.7"

while getopts s:p: option
do
 case "${option}"
 in
 s) SPARKNLP=${OPTARG};;
 p) PYSPARK=${OPTARG};;
 esac
done

echo "setup Colab for PySpark $PYSPARK and Spark NLP $SPARKNLP"
apt-get update
apt-get purge -y openjdk-11* -qq > /dev/null && sudo apt-get autoremove -y -qq > /dev/null
apt-get install -y openjdk-8-jdk-headless -qq > /dev/null

if [[ "$PYSPARK" == "3.1"* ]]; then
  wget -q "https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop2.7.tgz" > /dev/null
  tar -xvf spark-3.1.1-bin-hadoop2.7.tgz > /dev/null
  SPARKHOME="/content/spark-3.1.1-bin-hadoop2.7"
elif [[ "$PYSPARK" == "3.0"* ]]; then
  wget -q "https://downloads.apache.org/spark/spark-3.0.2/spark-3.0.2-bin-hadoop2.7.tgz" > /dev/null
  tar -xvf spark-3.0.2-bin-hadoop2.7.tgz > /dev/null
  SPARKHOME="/content/spark-3.0.2-bin-hadoop2.7"
elif [[ "$PYSPARK" == "2"* ]]; then
  wget -q "https://downloads.apache.org/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz" > /dev/null
  tar -xvf spark-2.4.7-bin-hadoop2.7.tgz > /dev/null
  SPARKHOME="/content/spark-2.4.7-bin-hadoop2.7"
else
  wget -q "https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop2.7.tgz" > /dev/null
  tar -xvf spark-3.1.1-bin-hadoop2.7.tgz > /dev/null
  SPARKHOME="/content/spark-3.1.1-bin-hadoop2.7"
fi

export SPARK_HOME=$SPARKHOME
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Install pyspark spark-nlp
! pip install --upgrade -q pyspark==$PYSPARK spark-nlp==$SPARKNLP findspark
