***************
Light Pipelines
***************

LightPipeline is a Spark NLP specific Pipeline class equivalent to the Spark ML
Pipeline. The difference is that it's execution does not hold to Spark principles,
instead it computes everything locally (but in parallel) in order to achieve fast
results when dealing with small amounts of data.

This means, we do not input a Spark Dataframe, but a string or an Array of strings
instead, to be annotated. To create Light Pipelines, you need to input an already
trained (fit) Spark ML Pipeline. It's ``transform()`` stage is converted into
``annotate()`` instead.

Converting PipelineModels
-------------------------

For example, the pipeline defined in the last section :doc:`custom_pipelines` can be
converted into a LightPipeline:

>>> from sparknlp.base import LightPipeline
>>> light = LightPipeline(pipeline.fit(data))
>>> light.annotate("We are very happy about Spark NLP")
{'token': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP']}


Pretrained Light Pipelines
--------------------------
PretrainedPipelines have a LightPipeline component and therefore have an interface for ``annotate()``:

>>> from sparknlp.pretrained import PretrainedPipeline
>>> explain_document_pipeline = PretrainedPipeline("explain_document_ml")
explain_document_ml download started this may take some time.
Approx size to download 9.1 MB
[OK!]
>>> explain_document_pipeline.annotate("We are very happy about Spark NLP")
{'document': ['We are very happy about Spark NLP'],
 'lemmas': ['We', 'be', 'very', 'happy', 'about', 'Spark', 'NLP'],
 'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP', 'NNP'],
 'sentence': ['We are very happy about Spark NLP'],
 'spell': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP'],
 'stems': ['we', 'ar', 'veri', 'happi', 'about', 'spark', 'nlp'],
 'token': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP']}