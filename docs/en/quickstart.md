---
layout: article
title: Quick Start
permalink: /docs/en/quickstart
key: docs-quickstart
modify_date: "2019-05-16"
---

# Spark-nlp 2.2.1 Quick Start Guideline

## The very first: Join our Slack channel

A good idea is to join our channel, to ask for help and share your feedback. Developers and users can help each other here getting started.

[Spark NLP Slack](https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLTM4ZDliMjU5OWZmMDE1ZGVkMjg0MWFjMjU3NjY4YThlMTJkNmNjNjM3NTMwYzlhMWY4MGMzODI2NDBkOWU4ZDE){:.button.button--info.button--rounded.button--md}

## The very second: Spark NLP Workshop

If you are of those who prefer learning by example, check this repository!

[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop){:.button.button--primary.button--rounded.button--md}

It is full of fresh examples and even a docker container if you want to skip installation.

Below, you can follow into a more theoretical and thorough quick start guide.

## Requirements & Setup

Spark NLP is built on top of **Apache Spark 2.4.0** and such is the **only** supported release.

It is recommended to have basic knowledge of the framework and a working environment before using Spark NLP. Refer to Spark [documentation](http://spark.apache.org/docs/2.4.0/index.html) to get started with Spark.

To start using the library, execute any of the following lines depending on your desired use case:

```bash
spark-shell --packages JohnSnowLabs:spark-nlp:2.2.1
pyspark --packages JohnSnowLabs:spark-nlp:2.2.1
spark-submit --packages JohnSnowLabs:spark-nlp:2.2.1
```

### **Straight forward Python on jupyter notebook**

As a first step we import the required python dependences including some sparknlp components.

Be sure that you have the required python libraries (pyspark 2.4.3, spark-nlp 2.2.1) by running pip list. Check that the versions are correct.

If some of them is missing you can run:

```bash
pip install --ignore-installed pyspark==2.4.3
pip install --ignore-installed spark-nlp==2.2.1
```

Of course you will need to have jupyter installed in your system:

```bash
pip install jupyter
```
Now you should be ready to create a jupyter notebook running from terminal:

```bash
jupyter notebook
```

The easiest way to get started, is to run the following code:

```python
import sparknlp
sparknlp.start()
```
After waiting some seconds you should see something like this in your notebook:

```python
SparkSession - in-memory

SparkContext

Spark UI

Version
    v2.4.3
Master
    local[*]
AppName
    Spark NLP
```

With those lines of code, you have successfully started a Spark Session and are ready to use Spark NLP

If you need more fine tuning, you will have to start SparkSession in your python program manually, this is an example

```python
spark = SparkSession.builder \
    .master('local[4]') \
    .appName('OCR Eval') \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.2.1") \
    .getOrCreate()
```

### **Python Jupyter Notebook with PySpark**

You can also run the Jupyter Notebook directly from Pyspark. In such a
case you don't need to open a session, it will be automatically started 
by pyspark. Just remember to setup the SPARK_HOME, PYSPARK_DRIVER_PYTHON and PYSPARK_DRIVER_PYTHON_OPTS


```python
export SPARK_HOME=/path/to/your/spark/folder
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook
```

To locate your SPARK_FOLDER you can for example run the following in a
linux system:

```bash
sudo find -wholename */jars/spark-core_*-2.4.3.jar
```
The parent folder where this ./jars/spark-core*-2.4.3.jar is your
SPARK_HOME folder.

**In Microsoft Windows systems you can search for that file location in 
the explorer.

Once you have setup those environmental variables you can start a jupyter
notebook with a Spark (including sparknlp) session directly opened by
running in your terminal:

```bash
pyspark --packages JohnSnowLabs:spark-nlp:2.2.1
```
### Spark-NLP from Scala

You can start a spark REPL with Scala by running in your terminal a
spark-shell including the JohnSnowLabs:spark-nlp:2.2.1 package:

```bash
spark-shell --packages JohnSnowLabs:spark-nlp:2.2.1
```


### Databricks cloud cluster & Apache Zeppelin

Add the following maven coordinates in the dependency configuration page:

```bash
com.johnsnowlabs.nlp:spark-nlp_2.11:2.2.1
```

For Python in **Apache Zeppelin** you may need to setup _**SPARK_SUBMIT_OPTIONS**_ utilizing --packages instruction shown above like this

```bash
export SPARK_SUBMIT_OPTIONS="--packages JohnSnowLabs:spark-nlp:2.2.1"
```


### S3 based standalone cluster (No Hadoop)

If your distributed storage is S3 and you don't have a standard hadoop configuration (i.e. fs.defaultFS) You need to specify where in the cluster distributed storage you want to store Spark NLP's tmp files. First, decide where you want to put your **application.conf** file

```bash
import com.johnsnowlabs.util.ConfigLoader
ConfigLoader.setConfigPath("/somewhere/to/put/application.conf")
```

And then we need to put in such application.conf the following content

```json
sparknlp {
    settings {
        cluster_tmp_dir = "somewhere in s3n:// path to some folder"
    }
}
```

For further alternatives and documentation check out our README page in [GitHub](https://github.com/JohnSnowLabs/spark-nlp).

## Concepts

Spark ML provides a set of Machine Learning applications, and it's logic consists of two main components: **Estimators** and **Transformers**. The first, have a method called fit() which secures and trains a piece of data to such application, and a **Transformer**, which is generally the result of a fitting process, applies changes to the the target dataset. These components have been embedded to be applicable to Spark NLP. **Pipelines** are a mechanism that allow multiple estimators and transformers within a single workflow, allowing multiple chained transformations along a Machine Learning task. Refer to [Spar kML](https://spark.apache.org/docs/2.3.0/ml-guide.html) library for more information.

## Annotation

An annotation is the basic form of the result of a Spark NLP operation. It's structure is made of:

- **annotatorType:** which annotator generated this annotation
- **begin:** the begin of the matched content relative to raw-text
- **end:** the end of the matched content relative to raw-text
- **result:** the main output of the annotation
- **metadata:** content of matched result and additional information
- **embeddings:** (new in 2.0) contains vector mappings if required

This object is **automatically generated** by annotators after a transform process. No manual work is required. But it must be understood in order to use it efficiently.

## Annotators

Annotators are the spearhead of NLP functions in Spark NLP. There are two forms of annotators:

- **Annotator Approaches:** Are those who represent a Spark ML Estimator and require a training stage. They have a function called fit(data) which trains a model based on some data. They produce the second type of annotator which is an annotator model or transformer.
- **Annotator Model:** They are spark models or transformers, meaning they have a **transform(data)** function which take a dataset and add to it a column with the result of this annotation. All transformers are additive, meaning they append to current data, never replace or delete previous information.

Both forms of annotators can be included in a Pipeline and will automatically go through all stages in the provided order and transform the data accordingly. A Pipeline is turned into a PipelineModel after the fit() stage. Either before or after can be saved and re-loaded to disk at any time.

### Common Functions

- **setInputCols**(column_names): Takes a list of column names of annotations required by this annotator
- **setOutputCol(**column_name): Defines the name of the column containing the result of this annotator. Use this name as an input for other annotators requiring the annotations of this one.

## Quickly annotate some text

You can run these examples using Python or Scala. 

The easiest way to run the python examples is by starting a pyspark
jupyter notebook including the spark-nlp package:

```bash
pyspark --packages JohnSnowLabs:spark-nlp:2.2.1
```

The easiest way of running these scala examples is by starting a
spark-shell session including the spark-nlp package:

```bash
spark-shell --packages JohnSnowLabs:spark-nlp:2.2.1
```

### Explain Document ML

Spark NLP offers a variety of pretrained pipelines that will help you
get started, and get a sense of how the library works. We are constantly
working on improving the available content.

### Downloading and using a pretrained pipeline

Explain Document ML, named as explain_document_ml is a pretrained
pipeline that does a little bit of everything NLP related. Let's try it
out in scala. Note that the first time might take longer since it
downloads the model from our servers!
 
#### Python code
```python
from sparknlp.pretrained import PretrainedPipeline
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
annotations = explain_document_pipeline.annotate("We are very happy about SparkNLP")
print(annotations)
```

```python
{
  'stem': ['we', 'ar', 'veri', 'happi', 'about', 'sparknlp'],
  'checked': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'lemma': ['We', 'be', 'very', 'happy', 'about', 'SparkNLP'],
  'document': ['We are very happy about SparkNLP'],
  'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP'],
  'token': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'sentence': ['We are very happy about SparkNLP']
}
```

#### Scala code
```scala
scala> import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
scala> val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
```
```
explain_document_ml download started this may take some time.
Approximate size to download 9.4 MB
Download done! Loading the resource.
explain_document_pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_ml,en,public/models)
```
```scala
val annotations = explainDocumentPipeline.annotate("We are very happy about SparkNLP")
println(annotations)
```
```
Map(
   stem -> List(we, ar, veri, happi, about, sparknlp), 
   checked -> List(We, are, very, happy, about, SparkNLP), 
   lemma -> List(We, be, very, happy, about, SparkNLP), 
   document -> List(We are very happy about SparkNLP), 
   pos -> ArrayBuffer(PRP, VBP, RB, JJ, IN, NNP), 
   token -> List(We, are, very, happy, about, SparkNLP), 
   sentence -> List(We are very happy about SparkNLP)
   )
```

As you can see the explain_document_ml is able to annotate any "document"
providing as output a list of stems, check-spelling, lemmas,
part of speech tags, tokens and sentence boundary detection and all this
"out-of-the-box"!.


### Using a pretrained pipeline with spark dataframes

You can also use the pipeline through a spark dataframe. You just need
to create first a spark dataframe with a column named "text" that will
work as the input for the pipeline and then use the .transform() method
to run the pipeline over that dataframe and store the outputs of the
different components in a spark dataframe.

Remember than when starting jupyter notebook from pyspark or when running
the spark-shell for scala a Spark Session is started in the background
by default within the namespace 'scala'.

#### Python code

```python
sentences = [
  ['Hello, this is an example sentence'],
  ['And this is a second sentence.']
]

# spark is the Spark Session automatically started by pyspark.
data = spark.createDataFrame(sentences).toDF("text")

# Download the pretrained pipeline from Johnsnowlab's servers
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
```
```
explain_document_ml download started this may take some time.
Approx size to download 9.4 MB
[OK!]
```
```python
# Transform 'data' and store output in a new 'annotations_df' dataframe
annotations_df = explain_document_pipeline.transform(data)

# Show the results
annotations_df.show()
```
```
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|             checked|               lemma|                stem|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|Hello, this is an...|[[document, 0, 33...|[[document, 0, 33...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, he...|[[pos, 0, 4, UH, ...|
|And this is a sec...|[[document, 0, 29...|[[document, 0, 29...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, an...|[[pos, 0, 2, CC, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
```

#### Scala code

```scala
scala> val data = Seq(
    "Hello, this is an example sentence",
    "And this is a second sentence")
    .toDF("text")
    
scala> data.show(truncate=false)
```
```
+------------------------------+
|text                          |
+------------------------------+
|Hello, this is an example set |
|And this is a second sentence.|
+------------------------------+
```

```scala
scala> val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
scala> val annotations_df = explainDocumentPipeline.transform(data)
scala> annotations_df.show()
```
```
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|             checked|               lemma|                stem|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|Hello, this is an...|[[document, 0, 33...|[[document, 0, 33...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, he...|[[pos, 0, 4, UH, ...|
|And this is a sec...|[[document, 0, 29...|[[document, 0, 29...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, an...|[[pos, 0, 2, CC, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
```
### Manipulating pipelines

The output of the previous DataFrame was in terms of Annotation objects.
 This output is not really confortable to deal with, as you can see by
running the code:

#### Python code
```python
annotations_df.select("token").show(truncate=False)
```
```
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|token                                                                                                                                                                                                                                                                                                                                       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[token, 0, 4, Hello, [sentence -> 0], [], []], [token, 5, 5, ,, [sentence -> 0], [], []], [token, 7, 10, this, [sentence -> 0], [], []], [token, 12, 13, is, [sentence -> 0], [], []], [token, 15, 16, an, [sentence -> 0], [], []], [token, 18, 24, example, [sentence -> 0], [], []], [token, 26, 33, sentence, [sentence -> 0], [], []]]|
|[[token, 0, 2, And, [sentence -> 0], [], []], [token, 4, 7, this, [sentence -> 0], [], []], [token, 9, 10, is, [sentence -> 0], [], []], [token, 12, 12, a, [sentence -> 0], [], []], [token, 14, 19, second, [sentence -> 0], [], []], [token, 21, 28, sentence, [sentence -> 0], [], []], [token, 29, 29, ., [sentence -> 0], [], []]]    |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

### Scala code
```scala
annotations_df.select("token").show(truncate=false)
```
```
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|token                                                                                                                                                                                                                                                                                                                                       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[token, 0, 4, Hello, [sentence -> 0], [], []], [token, 5, 5, ,, [sentence -> 0], [], []], [token, 7, 10, this, [sentence -> 0], [], []], [token, 12, 13, is, [sentence -> 0], [], []], [token, 15, 16, an, [sentence -> 0], [], []], [token, 18, 24, example, [sentence -> 0], [], []], [token, 26, 33, sentence, [sentence -> 0], [], []]]|
|[[token, 0, 2, And, [sentence -> 0], [], []], [token, 4, 7, this, [sentence -> 0], [], []], [token, 9, 10, is, [sentence -> 0], [], []], [token, 12, 12, a, [sentence -> 0], [], []], [token, 14, 19, second, [sentence -> 0], [], []], [token, 21, 28, sentence, [sentence -> 0], [], []], [token, 29, 29, ., [sentence -> 0], [], []]]    |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
 
What if we want to deal with just the
resulting annotations? We can use the Finisher annotator, retrieve the
Explain Document ML pipeline, and add them together in a Spark ML
Pipeline. Remember that pretrained pipelines expect the input column to be
named "text".

#### Python code
```python
from sparknlp import Finisher
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

finisher = Finisher().setInputCols(["token", "lemma", "pos"])
explain_pipeline_model = PretrainedPipeline("explain_document_ml").model

pipeline = Pipeline() \
    .setStages([
        explain_pipeline_model,
        finisher
        ])


sentences = [
    ['Hello, this is an example sentence'],
    ['And this is a second sentence.']
]
data = spark.createDataFrame(sentences).toDF("text")

model = pipeline.fit(data)
annotations_finished_df = model.transform(data)

annotations_finished_df.select('finished_token').show(truncate=False)
```

```
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[Hello, ,, this, is, an, example, sentence]|
|[And, this, is, a, second, sentence, .]    |
+-------------------------------------------+
```

#### Scala code
```scala
scala> import com.johnsnowlabs.nlp.Finisher
scala> import org.apache.spark.ml.Pipeline

scala> val finisher = new Finisher().setInputCols("token", "lemma", "pos")

scala> val explainPipelineModel = PretrainedPipeline("explain_document_ml").model

scala> val pipeline = new Pipeline().
    setStages(Array(
        explainPipelineModel,
        finisher
    ))

scala> val data = Seq(
    "Hello, this is an example sentence",
    "And this is a second sentence")
    .toDF("text")

scala> val model = pipeline.fit(data)
scala> val annotations_df = model.transform(data)
scala> annotations_df.select("finished_token").show(truncate=false)
```

```
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[Hello, ,, this, is, an, example, sentence]|
|[And, this, is, a, second, sentence, .]    |
+-------------------------------------------+
```

## Setup your own pipeline

### Annotator types

Every annotator has a type. Those annotators that share a type, can be
used interchangeably, meaning you could you use any of them when needed.

For example, when a token type annotator is required by another annotator,
such as a sentiment analysis annotator, you can either provide a normalized
token or a lemma, as both are of type token.

### Necessary imports

Since version 1.5.0 we are making necessary imports easy to reach,
**base.\_** will include general Spark NLP transformers and concepts,
while **annotator.\_** will include all annotators that we currently
provide. We also need Spark ML pipelines.

```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
```

```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
```

### DocumentAssembler: Getting data in

In order to get through the NLP process, we need to get raw data
annotated. There is a special **transformer** that does this for us:
the **DocumentAssembler**, it creates the first annotation of type
**Document** which may be used by annotators down the road

```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
```

```scala
val documentAssembler = new DocumentAssembler().
    setInputCol("text").
    setOutputCol("document")
```

### Sentence detection and tokenization

In this quick example, we now proceed to identify the sentences in each
of our document lines. SentenceDetector requires a Document annotation,
which is provided by the DocumentAssembler output, and it's itself a
Document type token. The Tokenizer requires a Document annotation type,
meaning it works both with DocumentAssembler or SentenceDetector output,
in here, we use the sentence output.

```python
sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("Sentence")
    
regexTokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
```

```scala
val sentenceDetector = new SentenceDetector().
    setInputCols(Array("document")).
    setOutputCol("sentence")

val regexTokenizer = new Tokenizer().
    setInputCols(Array("sentence")).
    setOutputCol("token")
```

We also include another special transformer, called **Finisher** to show
tokens in a human language.

```python
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setCleanAnnotations(False)
```

```scala
val finisher = new Finisher().
    setInputCols("token").
    setCleanAnnotations(false)
```

## Using Spark ML Pipeline

Now we want to put all this together and retrieve the results, we use a
Pipeline for this.  We use the same data in fit() that we will use in
transform since none of the pipeline stages have a training stage.

```python
pipeline = Pipeline() \
    .setStages([ 
        documentAssembler, 
        sentenceDetector, 
        regexTokenizer, 
        finisher 
    ])


```


```
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[hello, ,, this, is, an, example, sentence]|
+-------------------------------------------+
```

```scala

val pipeline = new Pipeline().
    setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher
    ))

scala> val data = Seq("hello, this is an example sentence").toDF("text")

scala> val annotations = pipeline.
    fit(data).
    transform(data).toDF("text"))
    
scala> annotations.select("finished_token").show(truncate=false)
```

```
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[hello, ,, this, is, an, example, sentence]|
+-------------------------------------------+
```

## Using Spark NLP's LightPipeline

LightPipeline is a Spark NLP specific Pipeline class equivalent to Spark
ML Pipeline. The difference is that it's execution does not hold to
Spark principles, instead it computes everything locally (but in
parallel) in order to achieve fast results when dealing with small
amounts of data. This means, we do not input a Spark Dataframe, but a 
string or an Array of strings instead, to be annotated. To create Light
Pipelines, you need to input an already trained (fit) Spark ML Pipeline.
It's transform() stage is converted into annotate() instead.

```scala
import com.johnsnowlabs.nlp.base._

val trainedModel = pipeline.fit(Seq.empty[String].toDF("text"))

val lightPipeline = new LightPipeline(trainedModel)

lightPipeline.annotate("Hello world, please annotate my text")
```

## Utilizing Spark NLP OCR Module

Spark NLP OCR Module is not included within Spark NLP. It is not an
annotator and not an extension to Spark ML. You can include it with the
following coordinates for Maven:

```bash
com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.2.1
```

### Creating Spark datasets from PDF (To be used with Spark NLP)

You can use OcrHelper to directly create spark dataframes from PDF.
This will hold entire documents in single rows, meant to be later
processed by a SentenceDetector. This way, you won't be breaking the
content in rows as if you were reading a standard document. Metadata
column will include page numbers and file name information per row.

```scala
import com.johnsnowlabs.nlp.util.io.OcrHelper

val myOcrHelper = new OcrHelper

val data = myOcrHelper.createDataset(spark, "/pdfs/")

val documentAssembler = new DocumentAssembler().setInputCol("text").setMetadataCol("metadata")

documentAssembler.transform(data).show()
```

### Creating an Array of Strings from PDF (For LightPipeline)

Another way, would be to simply create an array of strings. This is
useful for example if you are parsing a small amount of pdf files and
would like to use LightPipelines instead. See an example below.

```scala
import com.johnsnowlabs.nlp.util.io.OcrHelper

val myOcrHelper = new OcrHelper

val raw = myOcrHelper.createMap("/pdfs/")

val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

val lightPipeline = new LightPipeline(new Pipeline().setStages(Array(documentAssembler, sentenceDetector)).fit(Seq.empty[String].toDF("text")))

pipeline.annotate(raw.values.toArray)
```

## Training annotators

### Training methodology

Training your own annotators is the most key concept when dealing with
real life scenarios. Any of the annotators provided above, such as
pretrained pipelines and models, will rarely ever apply to a specific
use case. Dealing with real life problems will require training your own
models. In Spark NLP, training annotators will vary depending on the
annotators. Currently, we support three ways:

1. Most annotators are capable of training through the dataset passed to
**fit()** just as Spark ML does. Annotators that use the suffix
**Approach** are trainable annotators. Training from fit() is the
standard behavior in Spark ML. Annotators have different schema
requirements for training. Check the reference to see what are the
requirements of each annotators.

2. Training from an **external source**: Some of our annotators train
from an external file or folder passed to the annotator as a param.
You will see such ones as **setCorpus()** or **setDictionary()** param
setter methods, allowing you to configure the input to use. You can set
Spark NLP to read them as Spark datasets or LINE_BY_LINE which is
usually faster for small files.

3. Last but not least, some of our annotators are **Deep Learning**
based. These models may be trained with the standard AnnotatorApproach
API just like any other annotator. For more advanced users, we also
allow importing your own graphs or even training from Python and
converting them into an AnnotatorModel.

## Where to go next

### Documentation and reference

Detailed information about Spark NLP concepts, annotators and more may
be found [HERE](annotators)

### More examples in Scala and Python

We are working on examples to show you how the library may be used in
different scenarios, take a look at our examples repository, which also
includes a Docker image:

[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop){:.button.button--primary.button--rounded.button--md}
