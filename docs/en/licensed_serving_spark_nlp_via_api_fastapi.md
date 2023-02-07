---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Serving Spark NLP&#58 FastAPI
permalink: /docs/en/licensed_serving_spark_nlp_via_api_fastapi
key: docs-experiment_tracking
modify_date: "2022-02-18"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

This is the second article of the “Serving Spark NLP via API” series, showcasing how to serve Spark NLP using [FastAPI](https://fastapi.tiangolo.com/) and [LightPipelines](https://medium.com/spark-nlp/spark-nlp-101-lightpipeline-a544e93f20f1) for a quick inference. Don’t forget to check the other articles in this series, namely:

* How to serve Spark NLP using Microsoft [Synapse ML](https://microsoft.github.io/SynapseML/), available [here](https://nlp.johnsnowlabs.com/docs/en/serving_spark_nlp_via_api_synapseml).

* How to serve Spark NLP using [Databricks](https://databricks.com/) Jobs and [MLFlow](https://mlflow.org/) Rest APIs, available [here](https://nlp.johnsnowlabs.com/docs/en/serving_spark_nlp_via_api_databricks_mlflow).

</div><div class="h3-box" markdown="1">

## Background

[Spark NLP](https://towardsdatascience.com/introduction-to-spark-nlp-foundations-and-basic-components-part-i-c83b7629ed59) is a Natural Language Understanding Library built on top of Apache Spark, leveranging Spark MLLib pipelines, that allows you to run NLP models at scale, including SOTA Transformers. Therefore, it’s the only production-ready NLP platform that allows you to go from a simple PoC on 1 driver node, to scale to multiple nodes in a cluster, to process big amounts of data, in a matter of minutes.

Before starting, if you want to know more about all the advantages of using Spark NLP (as the ability to work at scale on [air-gapped environments](https://nlp.johnsnowlabs.com/docs/en/install#offline), for instance) we recommend you to take a look at the following resources:

* [John Snow Labs webpage](https://www.johnsnowlabs.com/);

* The official [technical documentation of Spark NLP](https://nlp.johnsnowlabs.com/);

* [Spark NLP channel on Medium](https://medium.com/spark-nlp);

* Also, follow [Veysel Kocaman](https://vkocaman.medium.com/), Data Scientist Lead and Head of Spark NLP for Healthcare, for the latests tips.

</div><div class="h3-box" markdown="1">

## Motivation

Spark NLP is server-agnostic, what means it does not come with an integrated API server, but offers a lot of options to serve NLP models using Rest APIs.

There is a wide range of possibilities to add a web server and serve Spark NLP pipelines using RestAPI, and in this series of articles we are only describing some of them.

Let’s have an overview of how to use Microsoft’s Synapse ML as an example for that purpose.

</div><div class="h3-box" markdown="1">

## FastAPI and Spark NLP LightPipelines

![Fast API serving of Spark NLP pipelines](https://cdn-images-1.medium.com/max/2046/1*du7p50wS_fIsaC_lR18qsg.png)

[FastAPI](https://fastapi.tiangolo.com/) is, as defined by the creators…
>  …a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

FastAPI provides with a very good latency and response times that, all along with the good performance of Spark NLP LightPipelines, makes this option the quickest one of the four described in the article.

Read more about the performance advantages of using *LightPipelines* in [this article](https://medium.com/spark-nlp/spark-nlp-101-lightpipeline-a544e93f20f1) created by John Snow Labs Data Scientist Lead [Veysel Kocaman](https://vkocaman.medium.com/).

</div><div class="h3-box" markdown="1">

### Strengths

* *Quickest approach*

* *Adds flexibility to build and adapt a custom API for your models*

</div><div class="h3-box" markdown="1">

### Weaknesses

* *LightPipelines are executed sequentially and don’t leverage the distributed computation that Spark Clusters provide.*

* *As an alternative, you can use FastAPI with default pipelines and a custom LoadBalancer, to distribute the calls over your cluster nodes.*

You can serve SparkNLP + FastAPI on Docker. To do that, we will create a project with the following files:

* **Dockerfile**: Image for creating a SparkNLP + FastAPI Docker image

* **requirements.txt**: PIP Requirements

* **entrypoint.sh**: Dockerfile entrypoint

* **content/**: folder containing FastAPI webapp and SparkNLP keys

* **content/main.py**: FastAPI webapp, entrypoint

* **content/sparknlp_keys.json**: SparkNLP keys (for Healthcare or OCR)

</div><div class="h3-box" markdown="1">

### Dockerfile

The aim of this file is to create a suitable Docker Image with all the OS and Python libraries required to run SparkNLP. Also, adds a entry endpoint for the FastAPI server (see below) and a main folder containing the actual code to run a pipeline on an input text and return the expected values.

    FROM ubuntu:18.04
    RUN apt-get update && apt-get -y update

    RUN apt-get -y update \
        && apt-get install -y wget \
        && apt-get install -y jq \
        && apt-get install -y lsb-release \
        && apt-get install -y openjdk-8-jdk-headless \
        && apt-get install -y build-essential python3-pip \
        && pip3 -q install pip --upgrade \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
             /usr/share/man /usr/share/doc /usr/share/doc-base

    ENV PYSPARK_DRIVER_PYTHON=python3
    ENV PYSPARK_PYTHON=python3

    ENV LC_ALL=C.UTF-8
    ENV LANG=C.UTF-8

    # We expose the FastAPI default port 8515
    EXPOSE 8515

    # Install all Python required libraries
    COPY requirements.txt /
    RUN pip install -r /requirements.txt

    # Adds the entrypoint to the FastAPI server
    COPY entrypoint.sh /
    RUN chmod +x /entrypoint.sh

    # In /content folder we will have our main.py and the license files
    COPY ./content/ /content/
    WORKDIR content/

    # We tell Docker to run this file when a container is instantiated
    ENTRYPOINT ["/entrypoint.sh"]

</div><div class="h3-box" markdown="1">

### requirements.txt

This file describes which Python libraries will be required when creating the Docker image to run Spark NLP on FastAPI.

    pyspark==3.1.2
    fastapi==0.70.1
    uvicorn==0.16
    wget==3.2
    pandas==1.4.1

</div><div class="h3-box" markdown="1">

### entrypoint.sh

This file is the entry point of our Docker container, which carries out the following actions:

 1. Takes the `sparknlp_keys.json` and exports its values as environment variables, as required by Spark NLP for Healthcare.

 2. Installs the proper version of Spark NLP for Healthcare, getting the values from the license keys we have just exported in the previous step.

 3. Runs the main.py file, that will load the pipelines and create and endpoint to serve them.

    ```
    #!/bin/bash

    # Load the license from sparknlp_keys.json and export the values as OS variables
    export_json () {
        for s in $(echo $values | jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]' $1 ); do
            export $s
        done
    }

    export_json "/content/sparknlp_keys.json"

    # Installs the proper version of Spark NLP for Healthcare
    pip install --upgrade spark-nlp-jsl==$JSL_VERSION --user --extra-index-url [https://pypi.johnsnowlabs.com/$SECRET](https://pypi.johnsnowlabs.com/$SECRET)

    if [ $? != 0 ];
    then
        exit 1
    fi

    # Script to create FastAPI endpoints and preloading pipelines for inference
    python3 /content/main.py
    ```

### *content/main.py*: Serving 2 pipelines in a FastAPI endpoint

To maximize the performance and minimize the latency, we are going to store two Spark NLP pipelines in memory, so that we load only once (at server start) and we just use them everytime we get an API request to infer.

To do this, let’s create a content/main.py Python script to download the required resources, store them in memory and serve them in Rest API endpoints.

First, the import section

    import uvicorn, json, os
    from fastapi import FastAPI
    from sparknlp.annotator import *
    from sparknlp_jsl.annotator import *
    from sparknlp.base import *
    import sparknlp, sparknlp_jsl
    from sparknlp.pretrained import PretrainedPipeline

    app = FastAPI()
    pipelines = {}

Then, let’s define the endpoint to serve the pipeline:

    @app.get("/benchmark/pipeline")
    async def get_one_sequential_pipeline_result(modelname, text=''):
        return pipelines[modelname].annotate(text)

Then, the startup event to preload the pipelines and start a Spark NLP Session:

    @app.on_event("startup")
    async def startup_event():
        with open('/content/sparknlp_keys.json', 'r') as f:
            license_keys = json.load(f)

    spark = sparknlp_jsl.start(secret=license_keys['SECRET'])

    pipelines['ner_profiling_clinical'] = PretrainedPipeline('ner_profiling_clinical', 'en', 'clinical/models')

    pipelines['clinical_deidentification'] = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

Finally, let’s run a uvicorn server, listening on port 8515 to the endpoints declared before:

    if __name__ == "__main__":
        uvicorn.run('main:app', host='0.0.0.0', port=8515)

### content/sparknlp_keys.json

For using Spark NLP for Healthcare, please add your Spark NLP for Healthcare license keys to content/sparknlp_keys.jsonDThe file is ready, you only need to fulfill with your own values taken from the json file John Snow Labs has provided you with.

    {
      "AWS_ACCESS_KEY_ID": "",
      "AWS_SECRET_ACCESS_KEY": "",
      "SECRET": "",
      "SPARK_NLP_LICENSE": "",
      "JSL_VERSION": "",
      "PUBLIC_VERSION": ""
    }

And now, let’s run the server!

 1. Creating the Docker image and running the container
    ```
    docker build -t johnsnowlabs/sparknlp:sparknlp_api .

    docker run -v jsl_keys.json:/content/sparknlp_keys.json -p 8515:8515 -it johnsnowlabs/sparknlp:sparknlp_api
    ```

2. Consuming the API using a Python script

Lets import some libraries

    import requests
    import time

Then, let’s create a clinical note

    ner_text = """
    A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO.
    He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
    """

We have preloaded and served two Pretrained Pipelines: clinical_deidentification and ner_profiling_clinical . In *modelname*, let’s set which one we want to check

    # Change this line to execute any of the two pipelines
    modelname = 'clinical_deidentification'
    # modelname = 'ner_profiling_clinical'

And finally, let’s use the requestslibrary to send a test request to the endpoint and get the results.

    query = f"?modelname={modelname}&text={ner_text}"
    url = f"http://localhost:8515/benchmark/pipeline{query}"

    print(requests.get(url))

Results (original and deidentified texts in json format)

    >> {
    'masked': ['A <AGE> female with a history of gestational diabetes mellitus diagnosed ...],

    'obfuscated': ['A 48 female with a history of gestational diabetes mellitus diagnosed ...'],

    'ner_chunk': ['28-year-old'],

    'sentence': ['A 28-year-old female with a history of gestational diabetes mellitus diagnosed ...']
    }

You can also prettify the json using the following function with the result of the `annotate()` function:

    def explode_annotate(ann_result):
       '''
       Function to convert result object to json
       input: raw result
       output: processed result dictionary
       '''
       result = {}
       for column, ann in ann_result[0].items():
           result[column] = []
           for lines in ann:
               content = {
                  "result": lines.result,
                  "begin": lines.begin,
                  "end": lines.end,
                  "metadata": dict(lines.metadata),
               }
               result[column].append(content)
       return result

</div><div class="h3-box" markdown="1">

## Do you want to know more?

* Check the example notebooks in the Spark NLP Workshop repository, available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/RestAPI)

* Visit [John Snow Labs](https://www.johnsnowlabs.com/) and [Spark NLP Technical Documentation](https://nlp.johnsnowlabs.com/) websites

* Follow us on Medium: [Spark NLP](https://medium.com/spark-nlp) and [Veysel Kocaman](https://vkocaman.medium.com/)

* Write to support@johnsnowlabs.com for any additional request you may have

</div>