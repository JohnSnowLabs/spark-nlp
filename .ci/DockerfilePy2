ARG from
FROM $from

RUN sbt assemblyAndCopy
RUN apt-get -y update && apt-get -y install python-pip
RUN pip2 install pyspark==2.3.3 numpy

WORKDIR python/
