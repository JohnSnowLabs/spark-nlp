FROM hseeberger/scala-sbt:8u212_1.2.8_2.13.0

COPY . /app/spark-nlp

ENV JAVA_OPTS="-Xmx2012m -XX:+UseG1GC"

WORKDIR /app/spark-nlp/

RUN sbt compile
