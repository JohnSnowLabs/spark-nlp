FROM hseeberger/scala-sbt

COPY . /app/spark-nlp

ENV JAVA_OPTS="-Xmx2012m -XX:+UseG1GC"

WORKDIR /app/spark-nlp/

RUN sbt compile
