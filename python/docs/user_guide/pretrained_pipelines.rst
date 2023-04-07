********************
Pretrained Pipelines
********************

Spark NLP offers a variety of pretrained pipelines that will help you get started, and get a sense
of how the library works. We are constantly working on improving the available content.

Downloading and using a pretrained pipeline
===========================================
In this example, Explain Document ML (``"explain_document_ml"``) is a pretrained pipeline that does a
little bit of everything NLP related.

Pretrained Pipelines can be used as a Spark ML Pipeline or a Spark NLP Light pipeline.

Note that the first time you run the below code it might
take longer since it downloads the pretrained pipeline from our servers!

>>> from sparknlp.pretrained import PretrainedPipeline
>>> explain_document_pipeline = PretrainedPipeline("explain_document_ml")
explain_document_ml download started this may take some time.
Approx size to download 9.1 MB
[OK!]


As a Spark ML Pipeline
----------------------
>>> data = spark.createDataFrame([["We are very happy about Spark NLP"]]).toDF("text")
>>> result = explain_document_pipeline.model.transform(data).selectExpr("explode(pos)")
>>> result.show()
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|               spell|              lemmas|               stems|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|We are very happy...|[[document, 0, 32...|[[document, 0, 32...|[[token, 0, 1, We...|[[token, 0, 1, We...|[[token, 0, 1, We...|[[token, 0, 1, we...|[[pos, 0, 1, PRP,...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+


As a Spark NLP LightPipeline
----------------------------

>>> explain_document_pipeline.annotate("We are very happy about Spark NLP")
{'document': ['We are very happy about Spark NLP'],
 'lemmas': ['We', 'be', 'very', 'happy', 'about', 'Spark', 'NLP'],
 'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP', 'NNP'],
 'sentence': ['We are very happy about Spark NLP'],
 'spell': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP'],
 'stems': ['we', 'ar', 'veri', 'happi', 'about', 'spark', 'nlp'],
 'token': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP']}


Available Pipelines
===================

Please see the `Pipelines Page <https://sparknlp.org/docs/en/pipelines>`_ for all available pipelines.

Alternatively you can also check for pretrained
pipelines using :meth:`.ResourceDownloader.showPublicPipelines`.

>>> ResourceDownloader.showPublicPipelines("en")
+------------------+------+---------+
| Pipeline         | lang | version |
+------------------+------+---------+
| dependency_parse | en   | 2.0.2   |
| check_spelling   | en   | 2.1.0   |
| match_datetime   | en   | 2.1.0   |
|  ...             | ...  | ...     |
