FROM openjdk:8-alpine

RUN apk add --no-cache --virtual=.dependencies tar wget bash rsync

ARG SBT_VERSION=0.13.16

RUN wget -qO- "https://cocl.us/sbt-$SBT_VERSION.tgz" \
    | tar xzf - -C /usr/local --strip-components=1 \
    && sbt exit

COPY . /app/spark-nlp

WORKDIR /app/spark-nlp/
RUN sbt publish-local
