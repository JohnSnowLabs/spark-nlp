FROM hseeberger/scala-sbt:8u252_1.3.12_2.12.11

COPY . /app/spark-nlp

ENV JAVA_OPTS="-Xmx2012m -XX:+UseG1GC"

WORKDIR /app/spark-nlp/

RUN sbt ++2.12.11 compile
