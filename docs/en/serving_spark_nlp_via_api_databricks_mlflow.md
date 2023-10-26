---
layout: docs
header: true
seotitle: Spark NLP - Serving with MLFlow on Databricks
title: Spark NLP - Serving with MLFlow on Databricks
permalink: /docs/en/serving_spark_nlp_via_api_databricks_mlflow
key: docs-experiment_tracking
modify_date: "2022-02-18"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

This is the first article of the “Serving Spark NLP via API” series, showcasing how to serve Spark NLP using [Databricks](https://databricks.com/) Jobs and [MLFlow](https://www.mlflow.org/) Serve APIs.

[//]: <> You can find two more approaches (first, using FastAPI and second, using SynapseML) in the Spark NLP for Healthcare [documentation](https://sparknlp.org/docs/en/licensed_install) page.

</div><div class="h3-box" markdown="1">

## Background

[Spark NLP](https://towardsdatascience.com/introduction-to-spark-nlp-foundations-and-basic-components-part-i-c83b7629ed59) is a Natural Language Understanding Library built on top of Apache Spark, leveranging Spark MLLib pipelines, that allows you to run NLP models at scale, including SOTA Transformers. Therefore, it’s the only production-ready NLP platform that allows you to go from a simple PoC on 1 driver node, to scale to multiple nodes in a cluster, to process big amounts of data, in a matter of minutes.

Before starting, if you want to know more about all the advantages of using Spark NLP (as the ability to work at scale on [air-gapped environments](https://sparknlp.org/docs/en/install#offline), for instance) we recommend you to take a look at the following resources:

* [John Snow Labs webpage](https://www.johnsnowlabs.com/);

* The official [technical documentation of Spark NLP](https://sparknlp.org/);

* [Spark NLP channel on Medium](https://medium.com/spark-nlp);

</div><div class="h3-box" markdown="1">

## Motivation

Spark NLP is server-agnostic, what means it does not come with an integrated API server, but offers a lot of options to serve NLP models using Rest APIs.

There is a wide range of possibilities to add a web server and serve Spark NLP pipelines using RestAPI, and in this series of articles we are only describing some of them.

Let’s have an overview of how to use Databricks Jobs API and MLFlow Serve as an example for that purpose.

</div><div class="h3-box" markdown="1">

## Databricks Jobs and MLFlow Serve APIs

![Using Databricks Jobs API to serve Spark NLP pipelines](https://cdn-images-1.medium.com/max/2000/1*c0GXoX-ad0SLbT_SuExJ9g.png)

</div><div class="h3-box" markdown="1">

### About Databricks

[Databricks ](http://databricks.com)is an enterprise software company founded by the creators of Apache Spark. The company has also created MLflow, the Serialization and Experiment tracking library you can use (inside or outside databricks), as described in the section “Experiment Tracking”.

Databricks develops a web-based platform for working with Spark, that provides automated cluster management and IPython-style notebooks. Their infrastructured is provided for training and production purposes, and is integrated in cloud platforms as Azure and AWS.

Spark NLP is a proud partner of Databricks and we offer a seamless integration with them — see [Install on Databricks](https://www.johnsnowlabs.com/databricks/). All Spark NLP capabilities run in Databricks, including MLFlow serialization and Experiment tracking, what can be used for serving Spark NLP for production purposes.

![Serving Spark NLP in Databricks with MLFlow](https://cdn-images-1.medium.com/max/2000/1*skSSkyIZTosgXlIRiz2ZTw.png)

</div><div class="h3-box" markdown="1">

### About MLFlow

[MLFlow ](http://mlflow.com)is a serialization and Experiment Tracking platform, which also natively suports Spark NLP. We have a documentation entry about MLFlow in the “Experiment Tracking” section. It’s highly recommended that you take a look before moving forward in this document, since we will use some of the concepts explained there.

We will use MLFlow serialization to serve our Spark NLP models.

</div><div class="h3-box" markdown="1">

### Strengths

* *Easily configurable and scalable clusters in Databricks*

* *Seamless integration of Spark NLP and Databricks for automatically creating Spark NLP clusters (check [Install on Databricks URL](https://www.johnsnowlabs.com/databricks/))*

* *Integration with MLFlow, experiment tracking, etc.*

* *Configure your training and serving environments separately. Use your serving environment for inference and scale it as you need.*

</div><div class="h3-box" markdown="1">

### Weaknesses

* *This approach does not allow you to customize your endpoints, it uses Databricks JOBS API ones*

* *Requires some time and expertise in Databricks to configure everything properly*

</div><div class="h3-box" markdown="1">

### Creating a cluster in Databricks

As mentioned before, Spark NLP offers a seamless integration with Databricks. To create a cluster, please follow the instructions in [Install on Databricks](https://www.johnsnowlabs.com/databricks/).

That cluster can be then replicated (cloned) for production purposes later on.

</div><div class="h3-box" markdown="1">

### Configuring Databricks for serving Spark NLP on MLFlow

In Databricks Runtime Version, select any Standard runtime, not ML ones... These add their version of MLFlow, and some incompatibilities may arise. For this example, we have used 8.3 (includes Apache Spark 3.1.1, Scala 2.12)

The cluster instantiated is prepared to use Spark NLP, but to make it production-ready using MLFlow, we need to add the MLFlow jar, in addition to the Spark NLP jar, as shown in the “Experiment Tracking” section.

In that case, we did it adding both jars...

```("spark.jars.packages":" com.johnsnowlabs.nlp:spark-nlp_2.12:[YOUR_SPARKNLP_VERSION],org.mlflow:mlflow-spark:1.21.0")```

...into the SparkSession. However, in Databricks, you don’t instantiate programmatically a session, but you configure it in the `Compute` screen, selecting your Spark NLP cluster, and then going to ```Configuration -> Advanced Options -> Spark -> Spark Config```, as shown in the following image:

![Adding the required jars to Spark Config](https://cdn-images-1.medium.com/max/2000/1*DAQ2fCmtwWJ0RsJn-gkkEQ.png)

In addition to Spark Config, we need to add the Spark NLP and MLFlow libraries to the Cluster. You can do that by going to Libraries inside your cluster. Make sure you have spark-nlp and mlflow. If not, you can install them either using PyPI or Maven artifacts. In the image below you can see the PyPI alternative:

![Adding PyPI dependencies to Libraries](https://cdn-images-1.medium.com/max/2000/1*qmIv1w2IUwbJDeUya6OwaA.png)

TIP: You can also use the Libraries section to add the jars (using Maven Coordinates) instead of setting them in the Spark Config, as showed before.

</div><div class="h3-box" markdown="1">

## Creating a notebook

You are ready to create a notebook in Databricks and attach it to the recently created cluster. To do that, go to `Create --> Notebook`, and select the cluster you want in the dropdown above your notebook. Make sure you have selected the cluster with the right Spark NLP + MLFlow configuration.

To check everything is ok, run the following lines:

1. To check the session is running:

    `spark`

2. To check jars are in the session:

    `spark.sparkContext.getConf().get('spark.jars.packages')`

You should see the following output from the last line (versions may differ depending on which ones you used to configure your cluster)

    Out[2]: 'com.johnsnowlabs.nlp:spark-nlp_2.12:[YOUR_SPARKNLP_VERSION],org.mlflow:mlflow-spark:1.21.0'

</div><div class="h3-box" markdown="1">

## Logging the experiment in Databricks using MLFlow

As explained in the “Experiment Tracking” section, MLFlow can log Spark MLLib / NLP Pipelines as experiments, to carry out runs on them, track versions, etc.

MLFlow is natively integrated in Databricks, so we can leverage the `mlflow.spark.log_model()` function of the Spark flavour of MLFlow, to start tracking our Spark NLP pipelines.

Let’s first import our libraries:

    import mlflow
    import sparknlp
    from sparknlp.base import *
    from sparknlp.annotator import *
    from pyspark.ml import Pipeline
    import pandas as pd
    from sparknlp.training import CoNLL
    import pyspark
    from pyspark.sql import SparkSession

Then, create a Lemmatization pipeline:

    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    lemmatizer = LemmatizerModel.pretrained() \
        .setInputCols(["token"]) \
        .setOutputCol("prediction")  # It's mandatory to call it prediction

    pipeline = Pipeline(stages=[
      documentAssembler,
      tokenizer,
      lemmatizer
     ])

    p_model = pipeline.fit( spark.createDataFrame([[""]]).toDF("text") )

IMPORTANT: Last output column of the last component in the pipeline should be called prediction.

Finally, let’s log the experiment. In the **Experiment Tracking** section, we used the pip_requirements parameter in the log_model() function to set the required libraries:

![Registration of a Spark NLP MLFlow experiment](https://cdn-images-1.medium.com/max/2198/1*WQl0tmVjuOsTP4GYBriBiA.png)

But we mentioned using conda is also available. Let’s use conda in this example:

    conda_env = {
        'channels': ['conda-forge'],
        'dependencies': [
            'python=3.8.8',
            {
                "pip": [
                  'pyspark==3.1.1',
                  'mlflow==1.21.0',
                  'spark-nlp==[YOUR_SPARKNLP_VERSION]'
                ]
            }
        ],
        'name': 'mlflow-env'
    }

With this conda environment, we are ready to log our pipeline:

    mlflow.spark.log_model(p_model, "lemmatizer", conda_env=conda_env)

You should see an output similar to this one:

    (6) Spark Jobs
    (1) MLflow run
    *Logged 1 run to an experiment in MLflow. Learn more*

</div><div class="h3-box" markdown="1">

## Experiment UI

On the top right corner of your notebook, you will see the Experiment widget, and inside, as shown in the image below.

![MLFlow Experiment UI](https://cdn-images-1.medium.com/max/2000/1*ZHic2YAYgZaX5I9p-UzFiw.png)

You can also access Experiments UI if you switch your environment from “Data Science & Engineering” to “Machine Learning”, on the left panel…

Once in the experiment UI, you will see the following screen, where your experiments are tracked.

![Experiments screen in MLFlow](https://cdn-images-1.medium.com/max/2066/1*dezCQLDphnjd7aGGdK456Q.png)

If you click on the Start Time cell of your experiment, you will reach the registered MLFlow run.

![MLFlow run screen](https://cdn-images-1.medium.com/max/2000/1*EfvX2LFkPq2fhXrxZ6L8jg.png)

On the left panel you will see the MLFlow model and some other artifacts, as the conda.yml and pip_requirements.txt that manage the dependencies of your models.

On the right panel, you will see two snippets, about how to call to the model for inference internally from Databricks.

 1. Snippet for calling with a Pandas Dataframe:
    ```
    import mlflow
    logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'
    ```

    ### Load model as a Spark UDF.
    `loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)`

    ### Predict on a Spark DataFrame.
    ```columns = list(df.columns)
    df.withColumn('predictions', loaded_model(*columns)).collect()
    ```

2. Snippet for calling with a Spark Dataframe. We won’t include it in this documentation because that snippet does not include SPark NLP specificities. To make it work, the correct snippet should be:

    ```import mlflow
    logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'
    loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)
   ```

    ### Predict on a Spark DataFrame.
    `res_spark = loaded_model.predict(df_1_spark.rdd)`

IMPORTANT: You will only get the last column (prediction) results, which is a list of Rows of Annotation Types. To convert the result list into a Spark Dataframe, use the following schema:

    import pyspark.sql.types as T
    import pyspark.sql.functions as f

    annotationType = T.StructType([
                T.StructField('annotatorType', T.StringType(), False),
                T.StructField('begin', T.IntegerType(), False),
                T.StructField('end', T.IntegerType(), False),
                T.StructField('result', T.StringType(), False),
                T.StructField('metadata', T.MapType(T.StringType(), T.StringType()), False),
                T.StructField('embeddings', T.ArrayType(T.FloatType()), False)
            ])

And then, get the results (for example, in res_spark) and apply the schema:

    spark_res = spark.createDataFrame(res_pandas[0], schema=annotationType)

</div><div class="h3-box" markdown="1">

## Calling the experiment for production purposes using MLFlow Rest API

Instead of choosing a Batch Inference, you can select REST API. This will lead you to another screen, when the model will be loaded for production purposes in an independent cluster. Once deployed, you will be able to:

1. Check the endpoint URL to consume the model externally;

2. Test the endpoint writing a json (in our example, ‘text’ is our first input col of the pipeline, so it shoud look similar to:

    `{"text": "This is a test of how the lemmatizer works"}`

You can see the response in the same screen.

3. Check what is the Python code or cURL command to do that very same thing programatically.

![Example of an official Databricks MLFlow Rest API example](https://cdn-images-1.medium.com/max/2048/0*b1TSvjP7CUj6to2m.gif)

By just using that Python code, you can already consume it for production purposes from any external web app.

_IMPORTANT: As per 17/02/2022, there is an issue being studied by Databricks team, regarding the creation on the fly of job clusters to serve MLFlow models that require configuring the Spark Session with specific jars. This will be fixed in later versions of Databricks. In the meantime, the way to go is using Databricks Jobs API._

</div><div class="h3-box" markdown="1">

## Calling the experiment for production purposes using Databricks Asynchronous Jobs API

### Creating the notebook for the inference job

And last, but not least, another approach to consume models for production purposes. the Jobs API.

Databricks has its own API for managing jobs, that allows you to instantiate any notebook or script as a job, run it, stop it, and manage all the life cycle. And you can configure the cluster where this job will run before hand, what prevents having the issue described in point 3.

To do that:

 1. Create a new production cluster, as described before, cloning you training environment but adapting it to your needs for production purposes. Make sure the Spark Config is right, as described at the beginning of this documentation.

 2. Create a new notebook. Always check that the jars are in the session:

    ```
    spark.sparkContext.getConf().get('spark.jars.packages')
    ```

    ```
    Out[2]: 'com.johnsnowlabs.nlp:spark-nlp_2.12:[YOUR_SPARKNLP_VERSION],org.mlflow:mlflow-spark:1.21.0'
    ```

 4. Add the Spark NLP imports.

    ```
     import mlflow
     import sparknlp
     from sparknlp.base import *
     from sparknlp.annotator import *
     from pyspark.ml import Pipeline
     import pandas as pd
     from sparknlp.training import CoNLL
     import pyspark
     from pyspark.sql import SparkSession
     import pyspark.sql.types as T
     import pyspark.sql.functions as f
     import json
    ```

 5. Let’s define that an input param called text will be sent in the request. Let’s get the text from that parameter using dbutils.
    ```
     input = ""
     try:
       input = dbutils.widgets.get("text")
       print('"text" input found: ' + input)
     except:
       print('Unable to run: dbutils.widgets.get("text"). Setting it to NOT_SET')
       input = "NOT_SET"
    ```

Right now, the input text will be in input var. You can trigger an exception or set the input to some default value if the parameter does not come in the request.

5. Let’s create a Spark Dataframe with the input

    `df = spark.createDataFrame([[input]]).toDF('text')`

6. And now, we just need to use the snippet for Spark Dataframe to consume MLFlow models, described above:

    ```
    import mlflow
    import pyspark.sql.types as T
    import pyspark.sql.functions as f

    logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'
    loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)
   ```

</div><div class="h3-box" markdown="1">

### Predict on a Spark DataFrame.
    res_spark = loaded_model.predict(df_1_spark.rdd)

    annotationType = T.StructType([
                T.StructField('annotatorType', T.StringType(), False),
                T.StructField('begin', T.IntegerType(), False),
                T.StructField('end', T.IntegerType(), False),
                T.StructField('result', T.StringType(), False),
                T.StructField('metadata', T.MapType(T.StringType(), T.StringType()), False),
                T.StructField('embeddings', T.ArrayType(T.FloatType()), False)
            ])

    spark_res = spark.createDataFrame(res_spark[0], schema=annotationType)

7. Let’s transform our lemmatized tokens from the Dataframe into a list of strings:
    ```
    lemmas = spark_res.select("result").collect()
    txt_results = [x['result'] for x in lemmas]
    ```

8. And finally, let’s use again dbutils to tell Databricks to spin off the run and return an exit parameter: the list of token strings.
    ```
    dbutils.notebook.exit(json.dumps({
      "status": "OK",
      "results": txt_results
    }))
   ```

</div><div class="h3-box" markdown="1">

### Configuring the job

Last, but not least. We need to precreate the job, so that we run it from the API. We could do that using the API as well, but we will show you how to do it using the UI.

On the left panel, go to Jobs and then Create Job.

![Configuring the job with a notebook and a cluster](https://cdn-images-1.medium.com/max/2000/1*XGxCvIy0JqPQElclU7gfZA.png)

In the jobs screen, you will see you job created. It’s not running, it’s prepared to be called on demand, programatically or in the interface, with a text input param. Let’s see how to do that:

</div><div class="h3-box" markdown="1">

### Running the job

 1. In the jobs screen, if you click on the job, you will enter the Job screen, and be able to set your text input parameter and run the job manually.

![Jobs screen in Databricks](https://cdn-images-1.medium.com/max/3664/1*-dayt-mDHanWLduP_nvkYg.png)

You can use this for testing purposes, but the interesting part is calling it externally, using the Databricks Jobs API.

2. Using the Databricks Jobs API, from for example, Postman.
    ```
    POST HTTP request
    URL: https://[your_databricks_instance]/api/2.1/jobs/run-now
    Authorization: [use Bearer Token. You can get it from Databricks, Settings, User Settings, Generate New Token.]
    Body:
    {
        "job_id": [job_id, check it in the Jobs screen],
        "notebook_params": {"text": "This is an example of how well the lemmatizer works"}
    }
    ```

As it’s an asynchronous call, it will return the number a number of run, but no results. You will need to query for results using the number of the run and the following url [https://[your_databricks_instance]/2.1/jobs/runs/get-output](https://[your_databricks_instance]/2.1/jobs/runs/get-output)

You will get a big json, but the most relevant info, the output, will be up to the end:

Results (list of lemmatized words)

    {"notebook_output": {
      "status": "OK",
      "results": ["This", "is", "a", "example", "of", "how", "lemmatizer", "work"]
    }}

The notebook will be prepared in the job, but idle, until you call it programatically, what will instantiate a run.

Check the Jobs [API](https://docs.databricks.com/dev-tools/api/latest/jobs.html) for more information about what you can do with it and how to adapt it to your solutions for production purposes.

</div><div class="h3-box" markdown="1">

## Do you want to know more?

[//]: <> Check how to productionize Spark NLP in our official documentation [here](https://sparknlp.org/docs/en/production-readiness)

* Visit [John Snow Labs](https://www.johnsnowlabs.com/) and [Spark NLP Technical Documentation](https://sparknlp.org/) websites

* Follow us on Medium: [Spark NLP](https://medium.com/spark-nlp) and [Veysel Kocaman](https://vkocaman.medium.com/)

* Write to support@johnsnowlabs.com for any additional request you may have

</div>