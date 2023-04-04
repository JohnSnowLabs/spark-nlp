---
layout: docs
header: true
seotitle: Spark NLP - Production
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

![](/assets/images/production-readiness/MicrosoftTeams-image.png)

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
<div class="h3-box" markdown="1">

## Productionizing Spark NLP using Synapse ML

![Rest API for John Snow Labs’ Spark NLP](https://cdn-images-1.medium.com/max/2118/1*I_lIG3imZDUAkS8aM4i4yA.png)

This is the first article of the “Serving Spark NLP via API” series, showcasing how to serve Sparkl NLP using Synapse ML and Fast API. There is another article in this series, that showcases how to serve Spark NLP using [Databricks ](https://databricks.com/)Jobs and [MLFlow ](https://mlflow.org/)Rest APIs, available here.

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

This is first of a series of 2 articles that explain four options you can use to serve Spark NLP models via Rest API:

 1. **Using Microsoft’s Synapse ML;**

 2. **Using FastAPI and LightPipelines;**

 3. Using Databricks Batch API (see Part 2/2 here);

 4. Using MLFlow serve API in Databricks (see Part 2/2 here);

All of them have their Strengths and weaknesses, so let’s go over them in detail.

</div><div class="h3-box" markdown="1">

## Microsoft’s Synapse ML

![SynapseML serving of Spark NLP pipelines](https://cdn-images-1.medium.com/max/5120/1*kMSVrOL2fTq4AX93lIOPZQ.jpeg)

[Synapse ML](https://microsoft.github.io/SynapseML/docs/about/) (previously named SparkMML) is, as they state in their official webpage:
>  … an ecosystem of tools aimed towards expanding the distributed computing framework [Apache Spark](https://github.com/apache/spark) in several new directions.

They offer a seamless integratation with OpenCV, LightGBM, Microsoft Cognitive Tool and, the most relevant for our use case, *Spark Serving*, an extension of *Spark Streaming *with an integrated server and a Load Balancer, that can attend multiple requests via Rest API, balance and attend them leveraging the capabilities of a Spark Cluster. That means that you can sin up a server and attend requests that will be distributed transparently over a Spark NLP cluster, in a very effortless way.

</div><div class="h3-box" markdown="1">

### Strengths

* *Ready-to-use server*

* *Includes a Load Balancer*

* *Distributes the work over a Spark Cluster*

* *Can be used for both Spark NLP and Spark OCR*

</div><div class="h3-box" markdown="1">

### Weaknesses

* *For small use cases that don’t require big cluster processing, other approaches may be faster (as FastAPI using LightPipelines)*

* *Requires using an external Framework*

* *This approach does not allow you to customize your endpoints, it uses Synapse ML ones*

</div><div class="h3-box" markdown="1">

### How to set up Synapse ML to serve Spark NLP pipelines

We will skip here how to install Spark NLP. If you need to do that, please follow this official webpage about how to install [Spark NLP](https://nlp.johnsnowlabs.com/docs/en/install) or, if [Spark NLP for Healthcare](https://nlp.johnsnowlabs.com/docs/en/licensed_install) if you are using the Healthcare library.

Synapse ML recommends using at least Spark 3.2, so first of all, let’s configure the Spark Session with the required jars packages(both for Synapse ML and Spark) with the the proper Spark version (take a look at the suffix spark-nlp-spark**32**) and also, very important, add to jars.repository the Maven repository for SynapseML.

    **sparknlpjsl_jar =** "spark-nlp-jsl.jar"

    **from** pyspark.sql **import** SparkSession

    **spark =** *SparkSession***.**builder \
        **.**appName("Spark") \
        **.**master("local[*]") \
        **.***config*("spark.driver.memory", "16G") \
        **.***config*("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        **.***config*("spark.kryoserializer.buffer.max", "2000M") \
        **.***config*("**spark.jars.packages**", "com.microsoft.azure:synapseml_2.12:0.9.5,com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:3.4.0")\
        **.***config*("**spark.jars**", sparknlpjsl_jar)\
        **.***config*("**spark.jars.repositories**", "https://mmlspark.azureedge.net/maven")\
        **.**getOrCreate()

After the initialization, add your required imports (Spark NLP) and add to them the SynapseML-specific ones:

    **import** sparknlp
    **import** sparknlp_jsl
    ...

    **import** synapse.ml
    **from** synapse.ml.io **import** *****

Now, let’s create a Spark NLP for Healthcare pipeline to carry out Entity Resolution.

    **document_assembler =** *DocumentAssembler*()\
          **.**setInputCol("text")\
          **.**setOutputCol("document")

    **sentenceDetectorDL =** *SentenceDetectorDLModel***.**pretrained("sentence_detector_dl_healthcare", "en", 'clinical/models') \
          **.**setInputCols(["document"]) \
          **.**setOutputCol("sentence")

    **tokenizer =** *Tokenizer*()\
          **.**setInputCols(["sentence"])\
          **.**setOutputCol("token")

    **word_embeddings =** *WordEmbeddingsModel***.**pretrained("embeddings_clinical", "en", "clinical/models")\
      **.**setInputCols(["sentence", "token"])\
      **.**setOutputCol("word_embeddings")

    **clinical_ner =** *MedicalNerModel***.**pretrained("ner_clinical", "en", "clinical/models") \
          **.**setInputCols(["sentence", "token", "word_embeddings"]) \
          **.**setOutputCol("ner")

    **ner_converter_icd =** *NerConverterInternal*() \
          **.**setInputCols(["sentence", "token", "ner"]) \
          **.**setOutputCol("ner_chunk")\
          **.**setWhiteList(['PROBLEM'])\
          **.**setPreservePosition(**False**)

    **c2doc =** *Chunk2Doc*()\
          **.**setInputCols("ner_chunk")\
          **.**setOutputCol("ner_chunk_doc")

    **sbert_embedder =** *BertSentenceEmbeddings***.**pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
          **.**setInputCols(["ner_chunk_doc"])\
          **.**setOutputCol("sentence_embeddings")\
          **.**setCaseSensitive(**False**)

    **icd_resolver =** *SentenceEntityResolverModel***.**pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
         **.**setInputCols(["ner_chunk", "sentence_embeddings"]) \
         **.**setOutputCol("icd10cm_code")\
         **.**setDistanceFunction("EUCLIDEAN")

    **resolver_pipeline =** *Pipeline*(
        stages **=** [
            document_assembler,
            sentenceDetectorDL,
            tokenizer,
            word_embeddings,
            clinical_ner,
            ner_converter_icd,
            c2doc,
            sbert_embedder,
            icd_resolver
      ])

Let’s use a clinical note to test Synapse ML.

    **clinical_note =** """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa; significantly, her abdominal examination was benign with no tenderness, guarding, or rigidity."""

Since SynapseML serves a RestAPI, we will be sending JSON requests. Let’s define a simple json with the clinical note:

    **data_json =** {"*text*": clinical_note }

Now, let’s spin up a server using Synapse ML Spark Serving. It will consist of:

 1. a streaming server that will receive a json and transform it into a Spark Dataframe

 2. a call to Spark NLP transform on the dataframe, using the pipeline

 3. a write operation returning the output also in json format.

    **#1: Creating the streaming server and transforming json to Spark Dataframe**
    **serving_input =** spark**.**readStream**.**server() \
        **.**address("localhost", 9999, "benchmark_api") \
        **.**option("name", "benchmark_api") \
        **.**load() \
        **.**parseRequest("benchmark_api", data**.**schema)


    **#2: Applying transform to the dataframe using our Spark NLP pipeline
    serving_output =** resolver_p_model**.**transform(serving_input) \
        **.**makeReply("icd10cm_code")


    **#3: Returning the response in json format**
    **server =** serving_output**.**writeStream \
          **.**server() \
          **.**replyTo("benchmark_api") \
          **.**queryName("benchmark_query") \
          **.**option("checkpointLocation", "file:///tmp/checkpoints-{}"**.**format(uuid**.**uuid1())) \
          **.**start()

And we are ready to test the endpoint using the requests library.

    **import** requests
    res **=** requests**.**post("http://localhost:9999/benchmark_api", data= json**.**dumps(data_json))

And last, but not least, let’s check the results:

    **for** i **in** range (0, len(response_list**.**json())):
      print(response_list**.**json()[i]['result'])

    >>O2441 O2411 P702 K8520 B159 E669 Z6841 R35 R631 R630 R111...

</div><div class="h3-box" markdown="1">

## Productionizing Spark NLP using FastAPI and LightPipelines

![Fast API serving of Spark NLP pipelines](https://cdn-images-1.medium.com/max/2046/1*du7p50wS_fIsaC_lR18qsg.png)

[FastAPI](https://fastapi.tiangolo.com/) is, as defined by the creators…
>  …a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

FastAPI provides with a very good latency and response times that, all along witht the good performance of Spark NLP LightPipelines, makes this option the quickest one of the four described in the article.

Read more about the performance advantages of using *LightPipelines *in [this article](https://medium.com/spark-nlp/spark-nlp-101-lightpipeline-a544e93f20f1) created by John Snow Labs Data Scientist Lead [Veysel Kocaman](https://vkocaman.medium.com/).

</div><div class="h3-box" markdown="1">

### Strengths

* *Quickest approach*

* *Adds flexibility to build and adapt a custom API for your models*

</div><div class="h3-box" markdown="1">

### **Weaknesses**

* *LightPipelines are executed sequentially and don’t leverage the distributed computation that Spark Clusters provide.*

* *As an alternative, you can use FastAPI with default pipelines and a custom LoadBalancer, to distribute the calls over your cluster nodes.*

You can serve SparkNLP + FastAPI on Docker. To do that, we will create a project with the following files:

* Dockerfile: Image for creating a SparkNLP + FastAPI Docker image

* requirements.txt: PIP Requirements

* entrypoint.sh: Dockerfile entrypoint

* content/: folder containing FastAPI webapp and SparkNLP keys

* content/main.py: FastAPI webapp, entrypoint

* content/sparknlp_keys.json: SparkNLP keys (for Healthcare or OCR)

</div><div class="h3-box" markdown="1">

### Dockerfile

The aim of this file is to create a suitable Docker Image with all the OS and Python libraries required to run SparkNLP. Also, adds a entry endpoint for the FastAPI server (see below) and a main folder containing the actual code to run a pipeline on an input text and return the expected values.

    **FROM **ubuntu:18.04
    **RUN **apt-get update && apt-get -y update

    **RUN **apt-get -y update \
        && apt-get install -y wget \
        && apt-get install -y jq \
        && apt-get install -y lsb-release \
        && apt-get install -y openjdk-8-jdk-headless \
        && apt-get install -y build-essential python3-pip \
        && pip3 -q install pip --upgrade \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
             /usr/share/man /usr/share/doc /usr/share/doc-base

    **ENV **PYSPARK_DRIVER_PYTHON=python3
    **ENV **PYSPARK_PYTHON=python3

    **ENV **LC_ALL=C.UTF-8
    **ENV **LANG=C.UTF-8

    **# We expose the FastAPI default port 8515**
    **EXPOSE **8515

    **# Install all Python required libraries**
    **COPY **requirements.txt /
    **RUN **pip install -r /requirements.txt

    **# Adds the entrypoint to the FastAPI server**
    **COPY **entrypoint.sh /
    **RUN **chmod +x /entrypoint.sh

    **# In /content folder we will have our main.py and the license files
    COPY **./content/ /content/
    **WORKDIR **content/

    **# We tell Docker to run this file when a container is instantiated**
    **ENTRYPOINT **["/entrypoint.sh"]

</div><div class="h3-box" markdown="1">

### requirements.txt

This file describes which Python libraries will be required when creating the Docker image to run Spark NLP on FastAPI.

    **pyspark**==3.1.2
    **fastapi**==0.70.1
    **uvicorn**==0.16
    **wget**==3.2
    **pandas**==1.4.1

</div><div class="h3-box" markdown="1">

### entrypoint.sh

This file is the entry point of our Docker container, which carries out the following actions:

 1. Takes the sparknlp_keys.json and exports its values as environment variables, as required by Spark NLP for Healthcare.

 2. Installs the proper version of Spark NLP for Healthcare, getting the values from the license keys we have just exported in the previous step.

 3. Runs the main.py file, that will load the pipelines and create and endpoint to serve them.

    *#!/bin/bash*

    **# Load the license from sparknlp_keys.json and export the values as OS variables
    ***export_json* () {
        for s in $(echo $values | jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]' $1 ); do
            export $s
        done
    }

    **export_json **"/content/sparknlp_keys.json"


    **# Installs the proper version of Spark NLP for Healthcare
    pip install **--upgrade spark-nlp-jsl==$JSL_VERSION --user --extra-index-url https://pypi.johnsnowlabs.com/$SECRET

    if [ $? != 0 ];
    then
        exit 1
    fi

    **# Script to create FastAPI endpoints and preloading pipelines for inference
    python3 **/content/main.py

***content/main.py*: Serving 2 pipelines in a FastAPI endpoint**

To maximize the performance and minimize the latency, we are going to store two Spark NLP pipelines in memory, so that we load only once (at server start) and we just use them everytime we get an API request to infer.

To do this, let’s create a **content/main.py** Python script to download the required resources, store them in memory and serve them in Rest API endpoints.

First, the import section

    **import** uvicorn, json, os
    **from** fastapi **import** FastAPI
    **from** sparknlp.annotator **import** *****
    **from **sparknlp_jsl.annotator **import *******
    **from** sparknlp.base **import** *****
    **import **sparknlp, sparknlp_jsl
    **from **sparknlp.pretrained **import** PretrainedPipeline

    app **=** FastAPI()
    pipelines **=** {}

Then, let’s define the endpoint to serve the pipeline:

    **@app.get("/benchmark/pipeline")**
    **async** **def** get_one_sequential_pipeline_result(modelname, text**=**''):
        **return** pipelines[modelname]**.**annotate(text)

Then, the startup event to preload the pipelines and start a Spark NLP Session:

    **@app.on_event("startup")**
    **async** **def** startup_event():
        **with** open('/content/sparknlp_keys.json', 'r') **as** f:
            license_keys **=** json**.**load(f)

     **   spark =** sparknlp_jsl**.**start(secret**=**license_keys['SECRE

        **pipelines**['ner_profiling_clinical'] **=** *PretrainedPipeline*('ner_profiling_clinical', 'en', 'clinical/models')

        **pipelines**['clinical_deidentification'] **=** *PretrainedPipeline*("clinical_deidentification", "en", "clinical/models")

Finally, let’s run a uvicorn server, listening on port 8515 to the endpoints declared before:

    **if __name__ == "__main__":**
        uvicorn**.**run('main:app', host**=**'0.0.0.0', port**=**8515)

**content/sparknlp_keys.json**

For using Spark NLP for Healthcare, please add your Spark NLP for Healthcare license keys to content/sparknlp_keys.jsonDThe file is ready, you only need to fulfill with your own values taken from the json file John Snow Labs has provided you with.

    {
      "**AWS_ACCESS_KEY_ID**": "",
      "**AWS_SECRET_ACCESS_KEY**": "",
      "**SECRET**": "",
      "**SPARK_NLP_LICENSE**": "",
      "**JSL_VERSION**": "",
      "**PUBLIC_VERSION**": ""
    }

And now, let’s run the server!

 1. **Creating the Docker image and running the container**

    **docker build** -t johnsnowlabs/sparknlp:sparknlp_api .

    **docker run **-v jsl_keys.json:/content/sparknlp_keys.json -p 8515:8515 -it johnsnowlabs/sparknlp:sparknlp_api

**2. Consuming the API using a Python script**

Lets import some libraries

    **import** requests
    **import** time

Then, let’s create a clinical note

    **ner_text =** """
    *A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO.
    He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.*
    """

We have preloaded and served two Pretrained Pipelines: clinical_deidentification and ner_profiling_clinical . In *modelname*, let’s set which one we want to check

    # Change this line to execute any of the two pipelines
    **modelname =** '*clinical_deidentification*'
    *# modelname = 'ner_profiling_clinical'*

And finally, let’s use the requestslibrary to send a test request to the endpoint and get the results.

    **query =** f"?modelname={modelname}&text={ner_text}"
    **url =** f"http://localhost:8515/benchmark/pipeline{query}"

    **print**(requests**.**get(url))

    >> {'sentence': ..., 'masked': ..., 'ner_chunk': ..., }

You can also prettify the json using the following function with the result of the annotate() function:

    **def explode_annotate(ann_result):**
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

</div>
