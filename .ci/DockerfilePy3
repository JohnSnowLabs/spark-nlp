ARG from
FROM $from

RUN sbt assemblyAndCopy
RUN apt-get -y update && apt-get -y install python3-pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install pyspark==2.4.0 numpy

ENV PYSPARK_PYTHON python3.6
ENV PYSPARK_DRIVER_PYTHON python3.6

WORKDIR python/
