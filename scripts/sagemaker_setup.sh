#!/bin/bash

# Default values for pyspark, spark-nlp, and SPARK_HOME
SPARKNLP="3.3.2"
PYSPARK="3.1.2"

SPARK_FOLDER_NAME="spark-$PYSPARK-bin-hadoop2.7"
SPARKHOME="/home/ec2-user/SageMaker/$SPARK_FOLDER_NAME"
APACHE_DL_URL="https://downloads.apache.org/spark"
TARGZ_URL_TO_DOWNLOAD="$APACHE_DL_URL/spark-$PYSPARK/$SPARK_FOLDER_NAME.tgz"

echo "Setup SageMaker for PySpark $PYSPARK and Spark NLP $SPARKNLP"
JAVA_8=$(alternatives --display java | grep 'jre-1.8.0-openjdk.x86_64/bin/java' | cut -d' ' -f1)
sudo alternatives --set java $JAVA_8

echo "Beginning download of Spark..."
wget -q "$APACHE_DL_URL/spark-$PYSPARK/$SPARK_FOLDER_NAME.tgz" > /dev/null
echo "Download done, will extract now."
tar -xvf "$SPARK_FOLDER_NAME.tgz" >/dev/null
echo "Spark has been downloaded and extracted."

export SPARK_HOME=$SPARKHOME

# Install pyspark spark-nlp
! pip install --upgrade -q pyspark==$PYSPARK spark-nlp==$SPARKNLP findspark