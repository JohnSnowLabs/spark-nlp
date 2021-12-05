---
layout: docs
header: true
seotitle: Spark NLP
title: Productionizing Spark NLP
permalink: /docs/en/production-readiness
key: docs-experiment_tracking
modify_date: "2021-11-21"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## Productionizing Spark NLP in Databricks
This documentation page will describe how to use Databricks to run Spark NLP Pipelines for production purposes.

</div><div class="h3-box" markdown="1">

## About Databricks
Databricks is an enterprise software company founded by the creators of Apache Spark. The company has also created MLflow, the Serialization and Experiment tracking library you can use (inside or outside databricks), as described in the section "Experiment Tracking".

Databricks develops a web-based platform for working with Spark, that provides automated cluster management and IPython-style notebooks. Their infrastructured is provided for training and production purposes, and is integrated in cloud platforms as Azure and AWS.

Spark NLP is a proud partner of Databricks and we offer a seamless integration with them - see [Install on Databricks](https://www.johnsnowlabs.com/databricks/). All Spark NLP capabilities run in Databricks, including MLFlow serialization and Experiment tracking, what can be used for serving Spark NLP for production purposes.

</div><div class="h3-box" markdown="1">

## About MLFlow
MLFlow is a serialization and Experiment Tracking platform, which also natively suports Spark NLP. We have a documentation entry about MLFlow in the "Experiment Tracking" section. It's highly recommended that you take a look before moving forward in this document, since we will use some of the concepts explained there.

We will use MLFlow serialization to serve our Spark NLP models.

</div><div class="h3-box" markdown="1">

## Creating a cluster in Databricks
As mentioned before, Spark NLP offers a seamless integration with Databricks. To create a cluster, please follow the instructions in [Install on Databricks](https://www.johnsnowlabs.com/databricks/).

That cluster can be then replicated (cloned) for production purposes later on.

</div><div class="h3-box" markdown="1">

## Configuring Databricks for Spark NLP and MLFlow
In `Databricks Runtime Version`, select any **Standard** runtime, **not ML** ones.. These ones add their version of MLFlow, and some incompatibilities may arise. For this example, we have used `8.3 (includes Apache Spark 3.1.1, Scala 2.12)`

The cluster instantiated is prepared to use Spark NLP, but to make it production-ready using MLFlow, we need to add the MLFlow jar, in addition to the Spark NLP jar, as shown in the "Experiment Tracking" section. 

In that case, we did it instantiating adding both jars (`"spark.jars.packages":" com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.2,org.mlflow:mlflow-spark:1.21.0"`) into the SparkSession. However, in Databricks, you don't instantiate programatically a session, but you configure it in the Compute screen, selecting your Spark NLP cluster, and then going to `Configuration -> Advanced Options -> Sparl -> Spark Config`, as shown in the following image:

![](/assets/images/production-readiness/db1.png)

In addition to Spark Config, we need to add the Spark NLP and MLFlow libraries to the Cluster. You can do that by going to `Libraries` inside your cluster. Make sure you have `spark-nlp` and `mlflow`. If not, you can install them either using PyPI or Maven artifacts. In the image below you can see the PyPI alternative:

![](/assets/images/production-readiness/db2.png)

</div><div class="h3-box" markdown="1">

## Creating a notebook
You are ready to create a notebook in Databricks and attach it to the recently created cluster. To do that, go to `Create - Notebook`, and select the cluster you want in the dropdown above your notebook. Make sure you have selected the cluster with the right Spark NLP + MLFlow configuration.

To check everything is ok, run the following lines:
1) To check the session is running:
```
spark
```
2) To check jars are in the session:
```
spark.sparkContext.getConf().get('spark.jars.packages')
```

You should see the following output from the last line (versions may differ depending on which ones you used to configure your cluster)
```
Out[2]: 'com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.2,org.mlflow:mlflow-spark:1.21.0'
```

</div><div class="h3-box" markdown="1">

## Logging the experiment in Databricks using MLFlow
As explained in the "Experiment Tracking" section, MLFlow can log Spark MLLib / NLP Pipelines as experiments, to carry out runs on them, track versions, etc.

MLFlow is natively integrated in Databricks, so we can leverage the `mlflow.spark.log_model()` function of the Spark flavour of MLFlow, to start tracking our Spark NLP pipelines.

Let's first import our libraries...
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
```

Then, create a Lemmatization pipeline:
```
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
```

IMPORTANT: Last output column of the last component in the pipeline should be called **prediction**.

Finally, let's log the experiment. In the "Experiment Tracking" section, we used the `pip_requirements` parameter in the `log_model()` function to set the required libraries:

![](/assets/images/production-readiness/db5.png)

But we mentioned using conda is also available. Let's use conda in this example:

```
conda_env = {
    'channels': ['conda-forge'],
    'dependencies': [
        'python=3.8.8',
        {
            "pip": [              
              'pyspark==3.1.1',
              'mlflow==1.21.0',
              'spark-nlp==3.3.2'
            ]
        }
    ],
    'name': 'mlflow-env'
}
```

With this conda environment, we are ready to log our pipeline:

```
mlflow.spark.log_model(p_model, "lemmatizer", conda_env=conda_env)
```

You should see an output similar to this one:
```
(6) Spark Jobs
(1) MLflow run
Logged 1 run to an experiment in MLflow. Learn more
```

</div><div class="h3-box" markdown="1">

## Experiment UI
On the top right corner of your notebook, you will see the Experiment widget, and inside, as shown in the image below.

![](/assets/images/production-readiness/db3.png)

You can also access Experiments UI if you switch your environment from "Data Science & Engineering" to "Machine Learning", on the left panel...

![](/assets/images/production-readiness/db4.png)


... or clicking on the "experiment" word in the cell output (it's a link!)

Once in the experiment UI, you will see the following screen, where your experiments are tracked.

![](/assets/images/production-readiness/db6.png)

If you click on the Start Time cell of your experiment, you will reach the registered MLFlow run.

![](/assets/images/production-readiness/db7.png)

On the left panel you will see the MLFlow model and some other artifacts, as the `conda.yml` and `pip_requirements.txt` that manage the dependencies of your models.

On the right panel, you will see two snippets, about how to call to the model for inference internally from Databricks.

1) Snippet for calling with a Pandas Dataframe:
```
import mlflow
logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
columns = list(df.columns)
df.withColumn('predictions', loaded_model(*columns)).collect()
```

2) Snippet for calling with a Spark Dataframe. We won't include it in this documentation because that snippet does not include SPark NLP specificities. To make it work, the correct snippet should be:
```
import mlflow
logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'

loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

# Predict on a Spark DataFrame.
res_spark = loaded_model.predict(df_1_spark.rdd)
```

IMPORTANT: You will only get the last column (`prediction`) results, which is a list of Rows of Annotation Types. To convert the result list into a Spark Dataframe, use the following schema:

```
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
```

And then, get the results (for example, in `res_spark`) and apply the schema:

```
spark_res = spark.createDataFrame(res_pandas[0], schema=annotationType)
```

</div><div class="h3-box" markdown="1">

## Calling the experiment for production purposes

### 1. Internally, if the data is in Databricks
If your data lies in Datalake, in Spark Tables, or any other internal storage in Databricks, you just need to use the previous snippets (depending if you want to use Pandas or Spark Dataframes), and you are ready to go. Example for Spark Dataframes:

![](/assets/images/production-readiness/db8.png)

Try to use Spark Dataframes by default, since converting from Spark Dataframes into Pandas triggers a `collect()` first, removing all the parallelism capabilities of Spark Dataframes.

The next logical step is to create Notebooks to be called programatically using the snippets above, running into production clusters. There are two ways to do this: using Batch Inference or using Jobs.

### 2. Internally, using Batch Inference (with Spark Tables)
If we come back to the experiment ui, you will see, above the Pandas and Spark snippets, a button with the text "Register Model". If you do that, you will register the experiment to be called externally, either for Batch Inference or with a REST API (we will get there!).

After clicking the Register Model button, you will see a link instead of the button, that will enabled after some seconds. By clicking that link, you will be redirected to the Model Inference screen.

![](/assets/images/production-readiness/db10.png)

This new screen has a button on the top right, that says "Use model for inference". By clicking on it, you will see two options: Batch Inference or REST API. Batch inference requires a Spark Table for input, and another for output, and after configuring them, what you will see is an auto-generated notebook to be executed on-demand, programatically or with crons, that is prepared to load the environment and do the inference, getting the text fron the input table and storing the results in the output table.

This is an example of how the notebook looks like:

![](/assets/images/production-readiness/db9.png)

### 3. Externally, with the MLFlow Serve REST API
Instead of chosing a Batch Inference, you can select REST API. This will lead you to another screen, when the model will be loaded for production purposes in an independent cluster. Once deployed, you will be able to:
1) Check the endpoint URL to consume the model externally;
2) Test the endpoint writing a json (in our example, 'text' is our first input col of the pipeline, so it shoud look similar to: 
```
{"text": "This is a test of how the lemmatizer works"}
```
You can see the response in the same screen.
3) Check what is the Python code or cURL command to do that very same thing programatically.

![](/assets/images/production-readiness/db12.gif)

By just using that Python code, you can already consume it for production purposes from any external web app.

IMPORTANT: As per 26/11/2021, there is an issue being studied by Databricks team, regarding the creation on the fly of job clusters to serve MLFlow models. There is not a way to configure the Spark Session, so the jars are not loaded and the model fails to start. This will be fixed in later versions of Databricks. In the meantime, see a workaround in point 4.

### 4. Databricks Jobs asynchronous REST API
#### Creating the notebook for the job
And last, but not least, another approach to consume models for production purposes. the Jobs API.

Databricks has its own API for managing jobs, that allows you to instantiate any notebook or script as a job, run it, stop it, and manage all the life cycle. And you can configure the cluster where this job will run before hand, what prevents having the issue described in point 3.

To do that:
1) Create a new production cluster, as described before, cloning you training environment but adapting it to your needs for production purposes. Make sure the Spark Config is right, as described at the beginning of this documentation.
2) Create a new notebook.
Always check that the jars are in the session:

```
spark.sparkContext.getConf().get('spark.jars.packages')
```
3) Add the Spark NLP imports.
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
4) Let's define that an input param called `text` will be sent in the request. Let's get the text from that parameter using `dbutils`.
```
input = ""
try:
  input = dbutils.widgets.get("text")
  print('"text" input found: ' + input)
except:
  print('Unable to run: dbutils.widgets.get("text"). Setting it to NOT_SET')
  input = "NOT_SET"
```

Right now, the input text will be in `input` var. You can trigger an exception or set the input to some default value if the parameter does not come in the request.

5) Let's create a Spark Dataframe with the input
```
df = spark.createDataFrame([[input]]).toDF('text')
```

6) And now, we just need to use the snippet for Spark Dataframe to consume MLFlow models, described above:
```
import mlflow
logged_model = 'runs:/a8cf070528564792bbf66d82211db0a0/lemmatizer'

loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

# Predict on a Spark DataFrame.
res_spark = loaded_model.predict(df_1_spark.rdd)

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
		
spark_res = spark.createDataFrame(res_spark[0], schema=annotationType)
```

7) Let's transform our lemmatized tokens from the Dataframe into a list of strings:
```
l = spark_res.select("result").collect()
txt_results = [x['result'] for x in l]
```

8) And finally, let's use again `dbutils` to tell Databricks to spin off the run and return an exit parameter: the list of token strings.

```
dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "results": txt_results
}))
```
#### Configuring the job 

Last, but not least. We need to precreate the job, so that we run it from the API. We could do that using the API as well, but we will show you how to do it using the UI.

On the left panel, go to `Jobs` and then `Create Job`.

![](/assets/images/production-readiness/db11.png)

In the jobs screen, you will see you job created. It's not running, it's prepared to be called on demand, programatically or in the interface, with a `text` input param. Let's see how to do that:

#### Running the job

1) In the jobs screen, if you click on the job, you will enter the Job screen, and be able to set your `text` input parameter and run the job manually.

![](/assets/images/production-readiness/db13.png)

You can use this for testing purpores, but the interesting part is calling it externally, using the Databricks Jobs API.

2) Using the Databricks Jobs API, from for example, Postman.
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

As it's an asynchronous call, it will return the number a number of run, but no results. You will need to query for results using the number of the run and the following url https://[your_databricks_instance]/2.1/jobs/runs/get-output

You will get a big json, but the most relevant info, the output, will be up to the end:

```
{"notebook_output": {
  "status": "OK",
  "results": ["This", "is", "a", "example", "of", "how", "lemmatizer", "work"]
}}
```

The notebook will be prepared in the job, but `idle`, until you call it programatically, what will instantiate a run.

Check the Jobs [API](https://docs.databricks.com/dev-tools/api/latest/jobs.html) for more information about what you can do with it and how to adapt it to your solutions for production purposes.

</div>