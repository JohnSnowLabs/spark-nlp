#!/bin/bash

#default values for pyspark, spark-nlp, and SPARK_HOME
SPARKNLP="4.0.0"
PYSPARK="3.2.1"

while getopts s:p: option
do
 case "${option}"
 in
 s) SPARKNLP=${OPTARG};;
 p) PYSPARK=${OPTARG};;
 esac
done

echo "setup Kaggle for PySpark $PYSPARK and Spark NLP $SPARKNLP"
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

if [[ "$PYSPARK" == "3.2"* ]]; then
  PYSPARK="3.2.1"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
  echo "Don't forget to use spark32=True inside sparknlp.start(spark32=True)"
elif [[ "$PYSPARK" == "3.1"* ]]; then
  PYSPARK="3.1.3"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
elif [[ "$PYSPARK" == "3.0"* ]]; then
  PYSPARK="3.0.3"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
else
  PYSPARK="3.0.3"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
fi

# Install pyspark spark-nlp
! pip install --upgrade -q pyspark==$PYSPARK spark-nlp==$SPARKNLP findspark
