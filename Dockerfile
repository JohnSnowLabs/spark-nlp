FROM openjdk:8-alpine

RUN apk add --no-cache --virtual=.dependencies tar wget bash

ARG SCALA_VERSION=2.11.11
ARG SBT_VERSION=0.13.13

RUN wget -qO- "http://downloads.lightbend.com/scala/$SCALA_VERSION/scala-$SCALA_VERSION.tgz" \
    | tar xzf - -C /usr/local --strip-components=1

RUN wget -qO- "http://dl.bintray.com/sbt/native-packages/sbt/$SBT_VERSION/sbt-$SBT_VERSION.tgz" \
    |  tar xzf - -C /usr/local --strip-components=1 \
    && sbt exit

COPY . /app/spark-nlp

WORKDIR /app/spark-nlp/
RUN sbt publish-local
