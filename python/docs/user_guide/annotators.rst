**********
Annotators
**********

Annotators are the spearhead of NLP functions in Spark NLP. Let's take the ``ClassifierDL``
Annotators as an example. There are two forms of annotators:

Annotator Approaches
====================

Annotator Approaches are those who represent a Spark ML Estimator and require a training stage.
They have a function called ``fit(data)`` which trains a model based on some data. They produce the
second type of annotator which is an annotator model or transformer.

**Example**

First we need to declare all the prerequisite steps and the training data:

>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> documentAssembler = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> useEmbeddings = UniversalSentenceEncoder.pretrained() \
...     .setInputCols(["document"]) \
...     .setOutputCol("sentence_embeddings")

In this example, the training data ``"sentiment.csv"`` has the form of::

    text,label
    This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
    This was a terrible movie! The acting was bad really bad!,1
    ...

and will be loaded with Spark:

>>> smallCorpus = spark.read.option("header","True").csv("src/test/resources/classifier/sentiment.csv")

Then we declare the :class:`.ClassifierDLApproach` that is going to be trained in the pipeline. Note that in this case,
the Annotator also requires a label column, set with ``setLabelColumn("label")``, to classify the text.

>>> docClassifier = ClassifierDLApproach() \
...     .setInputCols(["sentence_embeddings"]) \
...     .setOutputCol("category") \
...     .setLabelColumn("label") \
...     .setBatchSize(64) \
...     .setMaxEpochs(20) \
...     .setLr(5e-3) \
...     .setDropout(0.5)
>>> pipeline = Pipeline().setStages([
...     documentAssembler,
...     useEmbeddings,
...     docClassifier
... ])

Finally the data is fit to the pipeline and the Annotator is trained:

>>> pipelineModel = pipeline.fit(smallCorpus)

The result is a ``PipelineModel`` that can be used with ``transform(data)`` to classify sentiment.

Annotator Models
================

Annotator Models are Spark models or transformers, meaning they have a ``transform(data)`` function.
This function takes as input a dataframe to which it adds a new column containing the result of the
current annotation. All transformers are additive, meaning they append to current data, never replace
or delete previous information.

Both forms of annotators can be included in a Pipeline. All annotators included in a Pipeline will
be automatically executed in the defined order and will transform the data accordingly. A Pipeline
is turned into a ``PipelineModel`` after the ``fit()`` stage. The Pipeline can be saved to disk and re-loaded
at any time.

Note
----
The ``Model`` suffix is explicitly stated when the annotator is the result of a training process.
Some annotators, such as ``Tokenizer`` are transformers, but do not contain the word Model since
they are not
trained annotators.

**Example**

First we need to declare all the prerequisite steps:

>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> documentAssembler = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> sentence = SentenceDetector() \
...     .setInputCols("document") \
...     .setOutputCol("sentence")
>>> useEmbeddings = UniversalSentenceEncoder.pretrained() \
...    .setInputCols("document") \
...    .setOutputCol("sentence_embeddings")

Here we use a pretrained :class:`.ClassifierDLModel`. Your trained approach from the previous example could
also be inserted.

>>> sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm") \
...     .setInputCols("sentence_embeddings") \
...     .setOutputCol("sarcasm")
>>> pipeline = Pipeline().setStages([
...     documentAssembler,
...     sentence,
...     useEmbeddings,
...     sarcasmDL
... ])

Then we can create some data to classify and use ``transform(data)`` to get the results.

>>> data = spark.createDataFrame([
...     ["I'm ready!"],
...     ["If I could put into words how much I love waking up at 6 am on Mondays I would."]
... ]).toDF("text")
>>> result = pipeline.fit(data).transform(data)
>>> result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out") \
...     .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm") \
...     .show(truncate=False)
+-------------------------------------------------------------------------------+-------+
|sentence                                                                       |sarcasm|
+-------------------------------------------------------------------------------+-------+
|I'm ready!                                                                     |normal |
|If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
+-------------------------------------------------------------------------------+-------+

Pretrained Models
=================

Model annotators have a ``pretrained()`` on it's static object, to retrieve the public pre-trained
version of a model.

>>> import sparknlp
>>> from sparknlp.annotator import *
>>> classifierDL = ClassifierDLModel.pretrained() \
...     .setInputCols(["sentence_embeddings"]) \
...     .setOutputCol("classification")

``pretrained(name, language, extra_location)`` will by default, bring a default pre-trained model.
Sometimes we offer more than one model, in which case, you may have to use name, language or extra
location to download them.

For a complete list of available pretrained models, head to the `Spark NLP Models
<https://nlp.johnsnowlabs.com/models>`__. Alternatively you can also check for pretrained
models of a particular annotator using :meth:`.ResourceDownloader.showPublicModels`.

>>> ResourceDownloader.showPublicModels("ClassifierDLModel", "en")
+-------------------------+------+---------+
| Model                   | lang | version |
+-------------------------+------+---------+
| classifierdl_use_trec6  |  en  | 2.5.0   |
| classifierdl_use_trec50 |  en  | 2.5.0   |
| classifierdl_use_spam   |  en  | 2.5.3   |
| ...                     |  en  | ...     |



Common Functions
================
* ``setInputCols(column_names)``
    Takes a list of column names of annotations required by this
    annotator. Those are generated by the annotators which precede the current annotator in the
    pipeline.
* ``setOutputCol(column_name)``
    Defines the name of the column containing the result of the current
    annotator. Use this name as an input for other annotators down the pipeline requiring the outputs
    generated by the current annotator.

Available Annotators
====================
For all available Annotators refer to the full API reference :mod:`sparknlp.annotator`.

