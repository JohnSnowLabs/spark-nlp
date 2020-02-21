---
layout: article
title: Concepts
permalink: /docs/en/concepts
key: docs-concepts
modify_date: "2019-10-23"
use_language_switchter: "Python-Scala"

---

## Concepts

Spark ML provides a set of Machine Learning applications, and it's logic consists of two main components: **Estimators** and **Transformers**. The first, have a method called fit() which secures and trains a piece of data to such application, and a **Transformer**, which is generally the result of a fitting process, applies changes to the the target dataset. These components have been embedded to be applicable to Spark NLP. **Pipelines** are a mechanism that allow multiple estimators and transformers within a single workflow, allowing multiple chained transformations along a Machine Learning task. Refer to [Spark ML](https://spark.apache.org/docs/2.4.4/ml-guide.html) library for more information.

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
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
```

The easiest way of running these scala examples is by starting a
spark-shell session including the spark-nlp package:

```bash
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1
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

#### Python

```python
import sparknlp
sparknlp.start()

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

#### Scala

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
```

```bash
explain_document_ml download started this may take some time.
Approximate size to download 9.4 MB
Download done! Loading the resource.
explain_document_pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_ml,en,public/models)
```

```scala
val annotations = explainDocumentPipeline.annotate("We are very happy about SparkNLP")
println(annotations)
```

```bash
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

***Python code***

```python
import sparknlp
sparknlp.start()

sentences = [
  ['Hello, this is an example sentence'],
  ['And this is a second sentence.']
]

# spark is the Spark Session automatically started by pyspark.
data = spark.createDataFrame(sentences).toDF("text")

# Download the pretrained pipeline from Johnsnowlab's servers
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
```

```bash
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

```bash
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|             checked|               lemma|                stem|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|Hello, this is an...|[[document, 0, 33...|[[document, 0, 33...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, He...|[[token, 0, 4, he...|[[pos, 0, 4, UH, ...|
|And this is a sec...|[[document, 0, 29...|[[document, 0, 29...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, An...|[[token, 0, 2, an...|[[pos, 0, 2, CC, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
```

***Scala code***

```scala
val data = Seq(
    "Hello, this is an example sentence",
    "And this is a second sentence")
    .toDF("text")

data.show(truncate=false)
```

```bash
+------------------------------+
|text                          |
+------------------------------+
|Hello, this is an example set |
|And this is a second sentence.|
+------------------------------+
```

```scala
val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
val annotations_df = explainDocumentPipeline.transform(data)
annotations_df.show()
```

```bash
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

***Python code***

```python
annotations_df.select("token").show(truncate=False)
```

```bash+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|token                                                                                                                                                                                                                                                                                                                                       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[token, 0, 4, Hello, [sentence -> 0], [], []], [token, 5, 5, ,, [sentence -> 0], [], []], [token, 7, 10, this, [sentence -> 0], [], []], [token, 12, 13, is, [sentence -> 0], [], []], [token, 15, 16, an, [sentence -> 0], [], []], [token, 18, 24, example, [sentence -> 0], [], []], [token, 26, 33, sentence, [sentence -> 0], [], []]]|
|[[token, 0, 2, And, [sentence -> 0], [], []], [token, 4, 7, this, [sentence -> 0], [], []], [token, 9, 10, is, [sentence -> 0], [], []], [token, 12, 12, a, [sentence -> 0], [], []], [token, 14, 19, second, [sentence -> 0], [], []], [token, 21, 28, sentence, [sentence -> 0], [], []], [token, 29, 29, ., [sentence -> 0], [], []]]    |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

***Scala code***

```scala
annotations_df.select("token").show(truncate=false)
```

```bash
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

***Python code***

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

```bash
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[Hello, ,, this, is, an, example, sentence]|
|[And, this, is, a, second, sentence, .]    |
+-------------------------------------------+
```

***Scala code***

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

```bash
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

***Python code***

```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
```

***Scala code***

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

***Python code***

```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
```

***Scala code***

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

***Python code***

```python
sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("Sentence")

regexTokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
```

***Scala code***

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

***Python code***

```python
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setCleanAnnotations(False)
```

***Scala code***

```scala
val finisher = new Finisher().
    setInputCols("token").
    setCleanAnnotations(false)
```

### Finisher: Getting data out

At the end of each pipeline or any stage that was done by Spark NLP, you may want to get results out whether onto another pipeline or simply write them on disk. The `Finisher` annotator helps you to clean the metadata (if it's set to true) and output the results into an array:

```python
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setIncludeMetadata(True)
```

```scala
val finisher = new Finisher()
    .setInputCols("token")
    .setIncludeMetadata(true)
```

Or, you can use Apache Spark functions (SQL) to manipulate the output DataFrame. Here we combine the tokens and NER results together:

```scala
finisher.withColumn("newCol", explode(arrays_zip($"finished_token", $"finished_ner")))
```

If you need to have a flattened DataFrame (each sub-array in a new column) from any annotations other than struct type columns, you can use `explode` function from Spark SQL.

```python
import pyspark.sql.functions as F

df.withColumn("tmp", F.explode("chunk")).select("tmp.*")
```

```scala
import org.apache.spark.sql.functions._

df.withColumn("tmp", explode(col("chunk"))).select("tmp.*")
```

## Using Spark ML Pipeline

Now we want to put all this together and retrieve the results, we use a
Pipeline for this.  We use the same data in fit() that we will use in
transform since none of the pipeline stages have a training stage.

### Python code

***Python code***

```python
pipeline = Pipeline() \
    .setStages([
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher
    ])
```

```bash
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[hello, ,, this, is, an, example, sentence]|
+-------------------------------------------+
```

***Scala code***

```scala

val pipeline = new Pipeline().
    setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher
    ))

val data = Seq("hello, this is an example sentence").toDF("text")
val annotations = pipeline.
    fit(data).
    transform(data).toDF("text"))

annotations.select("finished_token").show(truncate=false)
```

```bash
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

***Python code***

```python
from pyspark.sql.types import StructType
emptyDataFrame = spark.createDataFrame([], StructType([]))
```

```bash
explain_document_ml download started this may take some time.
Approx size to download 9.4 MB
[OK!]
```

```python
lightPipeline = LightPipeline(explainDocumentPipeline.model)
lightPipeline.annotate("Hello world, please annotate my text")
```

```bash
{'stem': ['hello', 'world', ',', 'pleas', 'annot', 'my', 'text'],
 'checked': ['Hello', 'world', ',', 'please', 'annotate', 'my', 'text'],
 'lemma': ['Hello', 'world', ',', 'please', 'annotate', 'i', 'text'],
 'document': ['Hello world, please annotate my text'],
 'pos': ['UH', 'NN', ',', 'VB', 'NN', 'PRP$', 'NN'],
 'token': ['Hello', 'world', ',', 'please', 'annotate', 'my', 'text'],
 'sentence': ['Hello world, please annotate my text']}
```

***Scala code***

```scala
import com.johnsnowlabs.nlp.base._
val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
val lightPipeline = new LightPipeline(explainDocumentPipeline.model)
lightPipeline.annotate("Hello world, please annotate my text")
```

```bash
Map[String,Seq[String]] =
  Map(
    stem -> List(hello, world, ,, pleas, annot, my, text),
    checked -> List(Hello, world, ,, please, annotate, my, tex), 
    lemma -> List(Hello, world, ,, please, annotate, i, text),
    document -> List(Hello world, please annotate my text),
    pos -> ArrayBuffer(UH, NN, ,, VB, NN, PRP$, NN),
    token -> List(Hello, world, ,, please, annotate, my, text),
    sentence -> List(Hello world, please annotate my text)
    )
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


### Spark NLP Imports

We attempt making necessary imports easy to reach, **base** will include
general Spark NLP transformers and concepts, while **annotator** will
include all annotators that we currently provide. **embeddings** include
word embedding annotators. This does not include Spark imports.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.embeddings import *
```

```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
```

### Spark ML Pipelines

SparkML Pipelines are a uniform structure that helps creating and tuning
practical machine learning pipelines. Spark NLP integrates with them
seamlessly so it is important to have this concept handy. Once a
**Pipeline** is trained with **fit()**, this becomes a **PipelineModel**  

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from pyspark.ml import Pipeline
pipeline = Pipeline().setStages([...])
```

```scala
import org.apache.spark.ml.Pipeline
new Pipeline().setStages(Array(...))
```

### LightPipeline

LightPipelines are Spark ML pipelines converted into a single machine
but multithreaded task, becoming more than 10x times faster for smaller
amounts of data (small is relative, but 50k sentences is roughly a good
maximum). To use them, simply plug in a trained (fitted) pipeline.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.base import LightPipeline
LightPipeline(someTrainedPipeline).annotate(someStringOrArray)
```

```scala
import com.johnsnowlabs.nlp.LightPipeline
new LightPipeline(somePipelineModel).annotate(someStringOrArray))
```

**Functions:**

- annotate(string or string\[\]): returns dictionary list of annotation
results
- fullAnnotate(string or string\[\]): returns dictionary list of entire
annotations content

### RecursivePipeline

Recursive pipelines are SparkNLP specific pipelines that allow a Spark
ML Pipeline to know about itself on every Pipeline Stage task, allowing
annotators to utilize this same pipeline against external resources to
process them in the same way the user decides. Only some of our
annotators take advantage of this. RecursivePipeline behaves exactly
the same than normal Spark ML pipelines, so they can be used with the
same intention.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.annotator import *
recursivePipeline = RecursivePipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ])
```

```scala
import com.johnsnowlabs.nlp.RecursivePipeline
val recursivePipeline = new RecursivePipeline()
        .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ))
```

### Params and Features

#### Annotator parameters

SparkML uses ML Params to store pipeline parameter maps. In SparkNLP,
we also use Features, which are a way to store parameter maps that are
larger than just a string or a boolean. These features are serialized
as either Parquet or RDD objects, allowing much faster and scalable
annotator information. Features are also broadcasted among executors for
better performance.  
