FROM jupyter/pyspark-notebook:java-11.0.15

ARG SPARKNLP_VERSION=4.3.1
RUN pip install --no-cache-dir spark-nlp==${SPARKNLP_VERSION}

# Create a new user
ENV NB_USER=johnsnow
ENV CHOWN_HOME=yes
ENV CHOWN_HOME_OPTS="-R"

WORKDIR /home/${NB_USER}
