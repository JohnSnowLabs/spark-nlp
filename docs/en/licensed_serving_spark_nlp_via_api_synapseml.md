---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Serving Spark NLP&#58 SynapseML
permalink: /docs/en/licensed_serving_spark_nlp_via_api_synapseml
key: docs-experiment_tracking
modify_date: "2022-02-18"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

This is the first article of the “Serving Spark NLP via API” series, showcasing how to serve Spark NLP using Synapse ML

Don’t forget to check the other articles in this series, namely:

* How to server Spark NLP using [FastAPI](https://fastapi.tiangolo.com/) and [LightPipelines](https://medium.com/spark-nlp/spark-nlp-101-lightpipeline-a544e93f20f1), available [here](https://nlp.johnsnowlabs.com/docs/en/serving_spark_nlp_via_api_fastapi).

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

    sparknlpjsl_jar = "spark-nlp-jsl.jar"

    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("Spark") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5,com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:[YOUR_SPARKNLP_VERSION])\
        .config("spark.jars", sparknlpjsl_jar)\
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")\
        .getOrCreate()

After the initialization, add your required imports (Spark NLP) and add to them the SynapseML-specific ones:

    import sparknlp
    import sparknlp_jsl
    ...

    import synapse.ml
    from synapse.ml.io import *

Now, let’s create a Spark NLP for Healthcare pipeline to carry out Entity Resolution.

    document_assembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

    sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", 'clinical/models') \
          .setInputCols(["document"]) \
          .setOutputCol("sentence")

    tokenizer = Tokenizer()\
          .setInputCols(["sentence"])\
          .setOutputCol("token")

    word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("word_embeddings")

    clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
          .setInputCols(["sentence", "token", "word_embeddings"]) \
          .setOutputCol("ner")

    ner_converter_icd = NerConverterInternal() \
          .setInputCols(["sentence", "token", "ner"]) \
          .setOutputCol("ner_chunk")\
          .setWhiteList(['PROBLEM'])\
          .setPreservePosition(False)

    c2doc = Chunk2Doc()\
          .setInputCols("ner_chunk")\
          .setOutputCol("ner_chunk_doc")

    sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
          .setInputCols(["ner_chunk_doc"])\
          .setOutputCol("sentence_embeddings")\
          .setCaseSensitive(False)

    icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
         .setInputCols(["sentence_embeddings"]) \
         .setOutputCol("icd10cm_code")\
         .setDistanceFunction("EUCLIDEAN")

    resolver_pipeline = Pipeline(
        stages = [
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

    clinical_note = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa; significantly, her abdominal examination was benign with no tenderness, guarding, or rigidity."""

Since SynapseML serves a RestAPI, we will be sending JSON requests. Let’s define a simple json with the clinical note:

    data_json = {"text": clinical_note }

Now, let’s spin up a server using Synapse ML Spark Serving. It will consist of:

 1. a streaming server that will receive a json and transform it into a Spark Dataframe

 2. a call to Spark NLP transform on the dataframe, using the pipeline

 3. a write operation returning the output also in json format.

    #1: Creating the streaming server and transforming json to Spark Dataframe
    ```
    serving_input = spark.readStream.server() \
        .address("localhost", 9999, "benchmark_api") \
        .option("name", "benchmark_api") \
        .load() \
        .parseRequest("benchmark_api", data.schema)
    ```

    #2: Applying transform to the dataframe using our Spark NLP pipeline
    ```
    serving_output = resolver_p_model.transform(serving_input) \
        .makeReply("icd10cm_code")
    ```

    #3: Returning the response in json format
    ```
    server = serving_output.writeStream \
          .server() \
          .replyTo("benchmark_api") \
          .queryName("benchmark_query") \
          .option("checkpointLocation", "file:///tmp/checkpoints-{}".format(uuid.uuid1())) \
          .start()
    ```

And we are ready to test the endpoint using the `requests` library.

    import requests
    res = requests.post("http://localhost:9999/benchmark_api", data= json.dumps(data_json))

And last, but not least, let’s check the results:

    for i in range (0, len(response_list.json())):
      print(response_list.json()[i]['result'])

Results (list of ICD-10-CM codes from NER chunks)

    >> O2441 O2411 P702 K8520 B159 E669 Z6841 R35 R631 R630 R111...

</div><div class="h3-box" markdown="1">

## SynapseML on Databricks

You can also run the above code in Databricks. To do that, you only need to **remove** the **Creating a Spark Session**, since Databricks manages that session for you.

After we remove that part of the code from our notebook, we need to set the same configuration params in the Cluster Configuration, so that Databricks spins a cluster with the proper jars and config params (similarly to what we did programatically in *Creating a Spark Session above*, but using *Databricks UI*)

To do so, go to **Compute →Clusters** in Databricks and create a new cluster (name it, for instance, *Synapse*).

![Creating a new cluster called “Synapse”](https://cdn-images-1.medium.com/max/2000/1*njxtH-sr06pBXlDbx1gtnQ.png)

In your environment variables, as always, add the keys from your license in a *key=value* format

![](https://cdn-images-1.medium.com/max/2000/1*kpurIakeiyDiSmdpCVxtOw.png)

Then, in **Cluster → Libraries**, you need to install:

* **SynapseML jar** (Maven → com.microsoft.azure:synapseml_2.12:0.9.5)
* **Spark NLP jar** ( Maven →com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:[YOUR_SPARKNLP_VERSION])
* **Spark NLP wheel** (PyPi → spark-nlp==[YOUR_SPARKNLP_VERSION])
* If you are using Spark NLP for Healthcare
  * **Spark NLP for Healthcare jar**. Download the jar using the secret from your license, and then upload the jar to DBFS and add it in the Libraries section (DBFS/ADLS → *dbfs:/FileStore/johnsnowlabs/libs/spark_nlp_jsl_[YOUR_SPARKNLP_VERSION].jar)*
  * **Spark NLP for Healthcare wheel**. Same that with the jar. Download the jar using the secret from your license, and then upload the jar to DBFS and add it in the Libraries section (DBFS/ADLS → *dbfs:/FileStore/johnsnowlabs/libs/spark_nlp_jsl_[YOUR_SPARKNLP_VERSION].whl)*

---

And the rest of the code from the **Importing all the libraries** section and on remains exactly the same.

![Synapse ML on Databricks: results](https://cdn-images-1.medium.com/max/2000/1*aeb15y1EG4w58Tpm5rGwBw.png)

## Do you want to know more?

* Check the example notebooks in the Spark NLP Workshop repository, available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/RestAPI)

* Visit [John Snow Labs](https://www.johnsnowlabs.com/) and [Spark NLP Technical Documentation](https://nlp.johnsnowlabs.com/) websites

* Follow us on Medium: [Spark NLP](https://medium.com/spark-nlp) and [Veysel Kocaman](https://vkocaman.medium.com/)

* Write to support@johnsnowlabs.com for any additional request you may have

</div>