****************************
Setting up your own pipeline
****************************

Annotator types
===============

Every annotator has a type. Those annotators that share a type, can be used interchangeably,
meaning you could use any of them when needed.

For example, when a token type annotator is required by another annotator, such as a
sentiment analysis annotator, you can either provide a normalized token or a lemma, as both
are of type token.

Necessary imports
=================
:mod:`sparknlp.base` will include general Spark NLP transformers and concepts, while
:mod:`sparknlp.annotator` will include all annotators that we currently provide.
We also need Spark ML pipelines.

>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> spark = sparknlp.start()

Constructing the Pipeline
=========================

In this example a simple Tokenizer pipeline is constructed.

DocumentAssembler: Getting data in
----------------------------------
In order to get through the NLP process, we need to get raw data annotated. There is a
special transformer that does this for us: the DocumentAssembler, it creates the first
annotation of type Document which may be used by annotators down the road.

>>> documentAssembler = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")

Sentence detection and tokenization
-----------------------------------

In this quick example, we now proceed to identify the sentences in the input document.
SentenceDetector requires a Document annotation, which is provided by the DocumentAssembler
output, and it’s itself a Document type token. The Tokenizer requires a Document annotation type.
That means it works both with DocumentAssembler or SentenceDetector output. In the following
example we use the sentence output.

>>> sentenceDetector = SentenceDetector() \
...     .setInputCols(["document"]) \
...     .setOutputCol("sentence")
>>> regexTokenizer = Tokenizer() \
...     .setInputCols(["sentence"]) \
...     .setOutputCol("token")


Finisher: Getting data out
--------------------------

At the end of each pipeline or any stage that was done by Spark NLP, you may want to get results
out whether onto another pipeline or simply write them on disk. The Finisher annotator helps you
to clean the metadata (if it’s set to true) and output the results into an array:

>>> finisher = Finisher() \
...     .setInputCols(["token"]) \
...     .setIncludeMetadata(True)

Putting it all together as a Spark ML Pipeline
----------------------------------------------

Now we want to put all this together and retrieve the results, we use a Pipeline for this. We use
the same data in ``fit()`` that we will use in transform since none of the pipeline stages have a
training stage.

>>> pipeline = Pipeline().setStages([
...     documentAssembler,
...     sentenceDetector,
...     regexTokenizer,
...     finisher
... ])
>>> data = spark.createDataFrame([["We are very happy about Spark NLP"]]).toDF("text")
>>> pipeline.fit(data).transform(data).select("finished_token").show(truncate=False)
+-----------------------------------------+
|finished_token                           |
+-----------------------------------------+
|[We, are, very, happy, about, Spark, NLP]|
+-----------------------------------------+
