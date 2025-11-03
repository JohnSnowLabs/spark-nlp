#!/bin/bash

#default values for pyspark, spark-nlp, and SPARK_HOME
SPARKNLP="6.2.0"
PYSPARK="3.4.4"

while getopts s:p:g option; do
  case "${option}" in
  s) SPARKNLP=${OPTARG} ;;
  p) PYSPARK=${OPTARG} ;;
  g) GPU="true" ;;
  *)
    echo "Error: Invalid option -${OPTARG}" >&2
    exit 1
    ;;
  esac
done

# Java check and install if missing
if ! type -p java >/dev/null 2>&1; then
  echo "Java not found. Installing OpenJDK 11..."
  apt-get install -y -qq openjdk-11-jdk > /dev/null 2>&1
fi

export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
export PATH=$PATH:$JAVA_HOME/bin

if [[ "$PYSPARK" == "3.3"* ]]; then
  PYSPARK="3.3.4"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
elif [[ "$PYSPARK" == "3.2"* ]]; then
  PYSPARK="3.2.4"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
elif [[ "$PYSPARK" == "3.1"* ]]; then
  PYSPARK="3.1.3"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
elif [[ "$PYSPARK" == "3.0"* ]]; then
  PYSPARK="3.0.3"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
else
  PYSPARK="3.4.4"
  echo "Installing PySpark $PYSPARK and Spark NLP $SPARKNLP"
fi

echo "setup Colab for PySpark $PYSPARK and Spark NLP $SPARKNLP"

if [[ "$GPU" == "true" ]]; then
  echo "Upgrading libcudnn8 to 8.1.0 for GPU"
  apt install -qq --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2 -y &>/dev/null
fi

# Install pyspark spark-nlp
! pip install --upgrade -q pyspark==$PYSPARK spark-nlp==$SPARKNLP findspark
