#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Contains all the basic components to create a Spark NLP Pipeline.

This module contains basic transformers and extensions to the Spark Pipeline
interface. These are the :class:`LightPipeline` and :class:`RecursivePipeline`
which offer additional functionality.
"""

from abc import ABC

from pyspark import keyword_only
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import Pipeline, PipelineModel, Estimator, Transformer
from sparknlp.common import AnnotatorProperties
from sparknlp.internal import AnnotatorTransformer, RecursiveEstimator, RecursiveTransformer

from sparknlp.annotation import Annotation
import sparknlp.internal as _internal


class LightPipeline:
    """Creates a LightPipeline from a Spark PipelineModel.

    LightPipeline is a Spark NLP specific Pipeline class equivalent to Spark
    ML Pipeline. The difference is that it’s execution does not hold to
    Spark principles, instead it computes everything locally (but in
    parallel) in order to achieve fast results when dealing with small
    amounts of data. This means, we do not input a Spark Dataframe, but a
    string or an Array of strings instead, to be annotated. To create Light
    Pipelines, you need to input an already trained (fit) Spark ML Pipeline.

    It’s :meth:`.transform` has now an alternative :meth:`.annotate`, which
    directly outputs the results.

    Parameters
    ----------
    pipelineModel : :class:`pyspark.ml.PipelineModel`
        The PipelineModel containing Spark NLP Annotators
    parse_embeddings : bool, optional
        Whether to parse embeddings, by default False

    Notes
    -----
    Use :meth:`.fullAnnotate` to also output the result as
    :class:`.Annotation`, with metadata.

    Examples
    --------
    >>> from sparknlp.base import LightPipeline
    >>> light = LightPipeline(pipeline.fit(data))
    >>> light.annotate("We are very happy about Spark NLP")
    {
        'document': ['We are very happy about Spark NLP'],
        'lemmas': ['We', 'be', 'very', 'happy', 'about', 'Spark', 'NLP'],
        'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP', 'NNP'],
        'sentence': ['We are very happy about Spark NLP'],
        'spell': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP'],
        'stems': ['we', 'ar', 'veri', 'happi', 'about', 'spark', 'nlp'],
        'token': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP']
    }
    """

    def __init__(self, pipelineModel, parse_embeddings=False):
        self.pipeline_model = pipelineModel
        self._lightPipeline = _internal._LightPipeline(pipelineModel, parse_embeddings).apply()

    @staticmethod
    def _annotation_from_java(java_annotations):
        annotations = []
        for annotation in java_annotations:
            annotations.append(Annotation(annotation.annotatorType(),
                                          annotation.begin(),
                                          annotation.end(),
                                          annotation.result(),
                                          annotation.metadata(),
                                          annotation.embeddings
                                          )
                               )
        return annotations

    def fullAnnotate(self, target):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[Annotation]
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.fullAnnotate('U.N. official Ekeus heads for Baghdad.')
        >>> result[0].keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result[0]["ner"]
        [Annotation(named_entity, 0, 2, B-ORG, {'word': 'U.N'}),
        Annotation(named_entity, 3, 3, O, {'word': '.'}),
        Annotation(named_entity, 5, 12, O, {'word': 'official'}),
        Annotation(named_entity, 14, 18, B-PER, {'word': 'Ekeus'}),
        Annotation(named_entity, 20, 24, O, {'word': 'heads'}),
        Annotation(named_entity, 26, 28, O, {'word': 'for'}),
        Annotation(named_entity, 30, 36, B-LOC, {'word': 'Baghdad'}),
        Annotation(named_entity, 37, 37, O, {'word': '.'})]
        """
        result = []
        if type(target) is str:
            target = [target]
        for row in self._lightPipeline.fullAnnotateJava(target):
            kas = {}
            for atype, annotations in row.items():
                kas[atype] = self._annotation_from_java(annotations)
            result.append(kas)
        return result

    def annotate(self, target):
        """Annotates the data provided, extracting the results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[dict] or dict
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.annotate('U.N. official Ekeus heads for Baghdad.')
        >>> result.keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result["ner"]
        ['B-ORG', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O']
        """
        def reformat(annotations):
            return {k: list(v) for k, v in annotations.items()}

        annotations = self._lightPipeline.annotateJava(target)

        if type(target) is str:
            result = reformat(annotations)
        elif type(target) is list:
            result = list(map(lambda a: reformat(a), list(annotations)))
        else:
            raise TypeError("target for annotation may be 'str' or 'list'")

        return result

    def transform(self, dataframe):
        """Transforms a dataframe provided with the stages of the LightPipeline.

        Parameters
        ----------
        dataframe : :class:`pyspark.sql.DataFrame`
            The Dataframe to be transformed

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            The transformed DataFrame
        """
        return self.pipeline_model.transform(dataframe)

    def setIgnoreUnsupported(self, value):
        """Sets whether to ignore unsupported AnnotatorModels.

        Parameters
        ----------
        value : bool
            Whether to ignore unsupported AnnotatorModels.

        Returns
        -------
        LightPipeline
            The current LightPipeline
        """
        self._lightPipeline.setIgnoreUnsupported(value)
        return self

    def getIgnoreUnsupported(self):
        """Gets whether to ignore unsupported AnnotatorModels.

        Returns
        -------
        bool
            Whether to ignore unsupported AnnotatorModels.
        """
        return self._lightPipeline.getIgnoreUnsupported()


class RecursivePipeline(Pipeline, JavaEstimator):
    """Recursive pipelines are Spark NLP specific pipelines that allow a Spark
    ML Pipeline to know about itself on every Pipeline Stage task.

    This allows annotators to utilize this same pipeline against external
    resources to process them in the same way the user decides.

    Only some of the annotators take advantage of this. RecursivePipeline
    behaves exactly the same as normal Spark ML pipelines, so they can be used
    with the same intention.

    Examples
    --------
    >>> from sparknlp.annotator import *
    >>> from sparknlp.base import *
    >>> recursivePipeline = RecursivePipeline(stages=[
    ...     documentAssembler,
    ...     sentenceDetector,
    ...     tokenizer,
    ...     lemmatizer,
    ...     finisher
    ... ])
    """
    @keyword_only
    def __init__(self, *args, **kwargs):
        super(RecursivePipeline, self).__init__(*args, **kwargs)
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.RecursivePipeline", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _fit(self, dataset):
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, Estimator) or isinstance(stage, Transformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage))
        indexOfLastEstimator = -1
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                indexOfLastEstimator = i
        transformers = []
        for i, stage in enumerate(stages):
            if i <= indexOfLastEstimator:
                if isinstance(stage, Transformer):
                    transformers.append(stage)
                    dataset = stage.transform(dataset)
                elif isinstance(stage, RecursiveEstimator):
                    model = stage.fit(dataset, pipeline=PipelineModel(transformers))
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
                else:
                    model = stage.fit(dataset)
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
            else:
                transformers.append(stage)
        return PipelineModel(transformers)


class RecursivePipelineModel(PipelineModel):
    """Fitted RecursivePipeline.

    Behaves the same as a Spark PipelineModel does. Not intended to be
    initialized by itself. To create a RecursivePipelineModel please fit data to
    a :class:`.RecursivePipeline`.
    """
    def __init__(self, pipeline_model):
        super(PipelineModel, self).__init__()
        self.stages = pipeline_model.stages

    def _transform(self, dataset):
        for t in self.stages:
            if isinstance(t, HasRecursiveTransform):
                # drops current stage from the recursive pipeline within
                dataset = t.transform_recursive(dataset, PipelineModel(self.stages[:-1]))
            elif isinstance(t, AnnotatorProperties) and t.getLazyAnnotator():
                pass
            else:
                dataset = t.transform(dataset)
        return dataset


class HasRecursiveFit(RecursiveEstimator, ABC):
    """Properties for the implementation of the RecursivePipeline."""
    pass


class HasRecursiveTransform(RecursiveTransformer):
    """Properties for the implementation of the RecursivePipeline."""
    pass


class DocumentAssembler(AnnotatorTransformer):
    """Prepares data into a format that is processable by Spark NLP.

    This is the entry point for every Spark NLP pipeline. The
    `DocumentAssembler` can read either a ``String`` column or an
    ``Array[String]``. Additionally, :meth:`.setCleanupMode` can be used to
    pre-process the text (Default: ``disabled``). For possible options please
    refer the parameters section.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCol
        Input column name
    outputCol
        Output column name
    idCol
        Name of String type column for row id.
    metadataCol
        Name of Map type column with metadata information
    calculationsCol
        Name of float vector map column to use for embeddings and other
        representations.
    cleanupMode
        How to cleanup the document , by default disabled.
        Possible values: ``disabled, inplace, inplace_full, shrink, shrink_full,
        each, each_full, delete_full``

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from pyspark.ml import Pipeline
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library."]]).toDF("text")
    >>> documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> result = documentAssembler.transform(data)
    >>> result.select("document").show(truncate=False)
    +----------------------------------------------------------------------------------------------+
    |document                                                                                      |
    +----------------------------------------------------------------------------------------------+
    |[[document, 0, 51, Spark NLP is an open-source text processing library., [sentence -> 0], []]]|
    +----------------------------------------------------------------------------------------------+
    >>> result.select("document").printSchema()
    root
    |-- document: array (nullable = True)
    |    |-- element: struct (containsNull = True)
    |    |    |-- annotatorType: string (nullable = True)
    |    |    |-- begin: integer (nullable = False)
    |    |    |-- end: integer (nullable = False)
    |    |    |-- result: string (nullable = True)
    |    |    |-- metadata: map (nullable = True)
    |    |    |    |-- key: string
    |    |    |    |-- value: string (valueContainsNull = True)
    |    |    |-- embeddings: array (nullable = True)
    |    |    |    |-- element: float (containsNull = False)
    """

    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    idCol = Param(Params._dummy(), "idCol", "column for setting an id to such string in row", typeConverter=TypeConverters.toString)
    metadataCol = Param(Params._dummy(), "metadataCol", "String to String map column to use as metadata", typeConverter=TypeConverters.toString)
    calculationsCol = Param(Params._dummy(), "calculationsCol", "String to Float vector map column to use as embeddigns and other representations", typeConverter=TypeConverters.toString)
    cleanupMode = Param(Params._dummy(), "cleanupMode", "possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full", typeConverter=TypeConverters.toString)
    name = 'DocumentAssembler'

    @keyword_only
    def __init__(self):
        super(DocumentAssembler, self).__init__(classname="com.johnsnowlabs.nlp.DocumentAssembler")
        self._setDefault(outputCol="document", cleanupMode='disabled')

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the input column
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)

    def setIdCol(self, value):
        """Sets name of string type column for row id.

        Parameters
        ----------
        value : str
            Name of the Id Column
        """
        return self._set(idCol=value)

    def setMetadataCol(self, value):
        """Sets name for Map type column with metadata information.

        Parameters
        ----------
        value : str
            Name of the metadata column
        """
        return self._set(metadataCol=value)

    def setCalculationsCol(self, value):
        """Sets name of float vector map column to use for embeddings and other
        representations.

        Parameters
        ----------
        value : str
            Name of the calculations column
        """
        return self._set(metadataCol=value)

    def setCleanupMode(self, value):
        """Sets how to cleanup the document, by default disabled.
        Possible values: ``disabled, inplace, inplace_full, shrink, shrink_full,
        each, each_full, delete_full``

        Parameters
        ----------
        value : str
            Cleanup mode
        """
        if value.strip().lower() not in ['disabled', 'inplace', 'inplace_full', 'shrink', 'shrink_full', 'each', 'each_full', 'delete_full']:
            raise Exception("Cleanup mode possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full")
        return self._set(cleanupMode=value)


class TokenAssembler(AnnotatorTransformer, AnnotatorProperties):
    """This transformer reconstructs a ``DOCUMENT`` type annotation from tokens,
    usually after these have been normalized, lemmatized, normalized, spell
    checked, etc, in order to use this document annotation in further
    annotators. Requires ``DOCUMENT`` and ``TOKEN`` type annotations as input.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    preservePosition
        Whether to preserve the actual position of the tokens or reduce them to
        one space

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First, the text is tokenized and cleaned

    >>> documentAssembler = DocumentAssembler() \\
    ...    .setInputCol("text") \\
    ...    .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...    .setInputCols(["document"]) \\
    ...    .setOutputCol("sentences")
    >>> tokenizer = Tokenizer() \\
    ...    .setInputCols(["sentences"]) \\
    ...    .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...    .setInputCols(["token"]) \\
    ...    .setOutputCol("normalized") \\
    ...    .setLowercase(False)
    >>> stopwordsCleaner = StopWordsCleaner() \\
    ...    .setInputCols(["normalized"]) \\
    ...    .setOutputCol("cleanTokens") \\
    ...    .setCaseSensitive(False)

    Then the TokenAssembler turns the cleaned tokens into a ``DOCUMENT`` type
    structure.

    >>> tokenAssembler = TokenAssembler() \\
    ...    .setInputCols(["sentences", "cleanTokens"]) \\
    ...    .setOutputCol("cleanText")
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library for advanced natural language processing."]]) \\
    ...    .toDF("text")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentenceDetector,
    ...     tokenizer,
    ...     normalizer,
    ...     stopwordsCleaner,
    ...     tokenAssembler
    ... ]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.select("cleanText").show(truncate=False)
    +---------------------------------------------------------------------------------------------------------------------------+
    |cleanText                                                                                                                  |
    +---------------------------------------------------------------------------------------------------------------------------+
    |[[document, 0, 80, Spark NLP opensource text processing library advanced natural language processing, [sentence -> 0], []]]|
    +---------------------------------------------------------------------------------------------------------------------------+
    """

    name = "TokenAssembler"
    preservePosition = Param(Params._dummy(), "preservePosition", "whether to preserve the actual position of the tokens or reduce them to one space", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TokenAssembler, self).__init__(classname="com.johnsnowlabs.nlp.TokenAssembler")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPreservePosition(self, value):
        """Sets whether to preserve the actual position of the tokens or reduce
        them to one space.

        Parameters
        ----------
        value : str
            Name of the Id Column
        """
        return self._set(preservePosition=value)


class Doc2Chunk(AnnotatorTransformer, AnnotatorProperties):
    """Converts ``DOCUMENT`` type annotations into ``CHUNK`` type with the
    contents of a ``chunkCol``.

    Chunk text must be contained within input ``DOCUMENT``. May be either
    ``StringType`` or ``ArrayType[StringType]`` (using setIsArray). Useful for
    annotators that require a CHUNK type input.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    chunkCol
        Column that contains the string. Must be part of DOCUMENT
    startCol
        Column that has a reference of where the chunk begins
    startColByTokenIndex
        Whether start column is prepended by whitespace tokens
    isArray
        Whether the chunkCol is an array of strings, by default False
    failOnMissing
        Whether to fail the job if a chunk is not found within document.
        Return empty otherwise
    lowerCase
        Whether to lower case for matching case

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.common import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> chunkAssembler = Doc2Chunk() \\
    ...     .setInputCols("document") \\
    ...     .setChunkCol("target") \\
    ...     .setOutputCol("chunk") \\
    ...     .setIsArray(True)
    >>> data = spark.createDataFrame([[
    ...     "Spark NLP is an open-source text processing library for advanced natural language processing.",
    ...     ["Spark NLP", "text processing library", "natural language processing"]
    ... ]]).toDF("text", "target")
    >>> pipeline = Pipeline().setStages([documentAssembler, chunkAssembler]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.selectExpr("chunk.result", "chunk.annotatorType").show(truncate=False)
    +-----------------------------------------------------------------+---------------------+
    |result                                                           |annotatorType        |
    +-----------------------------------------------------------------+---------------------+
    |[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
    +-----------------------------------------------------------------+---------------------+

    See Also
    --------
    Chunk2Doc : for converting `CHUNK` annotations to `DOCUMENT`
    """

    chunkCol = Param(Params._dummy(), "chunkCol", "column that contains string. Must be part of DOCUMENT", typeConverter=TypeConverters.toString)
    startCol = Param(Params._dummy(), "startCol", "column that has a reference of where chunk begins", typeConverter=TypeConverters.toString)
    startColByTokenIndex = Param(Params._dummy(), "startColByTokenIndex", "whether start col is by whitespace tokens", typeConverter=TypeConverters.toBoolean)
    isArray = Param(Params._dummy(), "isArray", "whether the chunkCol is an array of strings", typeConverter=TypeConverters.toBoolean)
    failOnMissing = Param(Params._dummy(), "failOnMissing", "whether to fail the job if a chunk is not found within document. return empty otherwise", typeConverter=TypeConverters.toBoolean)
    lowerCase = Param(Params._dummy(), "lowerCase", "whether to lower case for matching case", typeConverter=TypeConverters.toBoolean)
    name = "Doc2Chunk"

    @keyword_only
    def __init__(self):
        super(Doc2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.Doc2Chunk")
        self._setDefault(
            isArray=False
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setChunkCol(self, value):
        """Sets column that contains the string. Must be part of DOCUMENT.

        Parameters
        ----------
        value : str
            Name of the Chunk Column
        """
        return self._set(chunkCol=value)

    def setIsArray(self, value):
        """Sets whether the chunkCol is an array of strings.

        Parameters
        ----------
        value : bool
            Whether the chunkCol is an array of strings
        """
        return self._set(isArray=value)

    def setStartCol(self, value):
        """Sets column that has a reference of where chunk begins.

        Parameters
        ----------
        value : str
            Name of the reference column
        """
        return self._set(startCol=value)

    def setStartColByTokenIndex(self, value):
        """Sets whether start column is prepended by whitespace tokens.

        Parameters
        ----------
        value : bool
            whether start column is prepended by whitespace tokens
        """
        return self._set(startColByTokenIndex=value)

    def setFailOnMissing(self, value):
        """Sets whether to fail the job if a chunk is not found within document.
        Return empty otherwise.

        Parameters
        ----------
        value : bool
            Whether to fail job on missing chunks
        """
        return self._set(failOnMissing=value)

    def setLowerCase(self, value):
        """Sets whether to lower case for matching case.

        Parameters
        ----------
        value : bool
            Name of the Id Column
        """
        return self._set(lowerCase=value)


class Chunk2Doc(AnnotatorTransformer, AnnotatorProperties):
    """Converts a ``CHUNK`` type column back into ``DOCUMENT``. Useful when
    trying to re-tokenize or do further analysis on a ``CHUNK`` result.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.pretrained import PretrainedPipeline

    Location entities are extracted and converted back into ``DOCUMENT`` type for
    further processing.

    >>> data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

    Define pretrained pipeline that extracts Named Entities amongst other things
    and apply `Chunk2Doc` on it.

    >>> pipeline = PretrainedPipeline("explain_document_dl")
    >>> chunkToDoc = Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
    >>> explainResult = pipeline.transform(data)

    Show results.

    >>> result = chunkToDoc.transform(explainResult)
    >>> result.selectExpr("explode(chunkConverted)").show(truncate=False)
    +------------------------------------------------------------------------------+
    |col                                                                           |
    +------------------------------------------------------------------------------+
    |[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
    |[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
    +------------------------------------------------------------------------------+

    See Also
    --------
    Doc2Chunk : for converting `DOCUMENT` annotations to `CHUNK`
    """

    name = "Chunk2Doc"

    @keyword_only
    def __init__(self):
        super(Chunk2Doc, self).__init__(classname="com.johnsnowlabs.nlp.Chunk2Doc")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


class Finisher(AnnotatorTransformer):
    """Converts annotation results into a format that easier to use.

    It is useful to extract the results from Spark NLP Pipelines. The Finisher
    outputs annotation(s) values into ``String``.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``ANY``                ``NONE``
    ====================== ======================

    Parameters
    ----------
    inputCols
        Input annotations
    outputCols
        Output finished annotation cols
    valueSplitSymbol
        Character separating values, by default #
    annotationSplitSymbol
        Character separating annotations, by default @
    cleanAnnotations
        Whether to remove annotation columns, by default True
    includeMetadata
        Whether to include annotation metadata, by default False
    outputAsArray
        Finisher generates an Array with the results instead of string, by
        default True
    parseEmbeddingsVectors
        Whether to include embeddings vectors in the process, by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

    Define pretrained pipeline that extracts Named Entities amongst other things
    and apply the `Finisher` on it.

    >>> pipeline = PretrainedPipeline("explain_document_dl")
    >>> finisher = Finisher().setInputCols("entities").setOutputCols("output")
    >>> explainResult = pipeline.transform(data)

    Show results.

    >>> explainResult.selectExpr("explode(entities)").show(truncate=False)
    +------------------------------------------------------------------------------------------------------------------------------------------------------+
    |entities                                                                                                                                              |
    +------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[[chunk, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []], [chunk, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]]|
    +------------------------------------------------------------------------------------------------------------------------------------------------------+
    >>> result = finisher.transform(explainResult)
    >>> result.select("output").show(truncate=False)
    +----------------------+
    |output                |
    +----------------------+
    |[New York, New Jersey]|
    +----------------------+

    See Also
    --------
    Finisher : for finishing Strings
    """

    inputCols = Param(Params._dummy(), "inputCols", "input annotations", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output finished annotation cols", typeConverter=TypeConverters.toListString)
    valueSplitSymbol = Param(Params._dummy(), "valueSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    annotationSplitSymbol = Param(Params._dummy(), "annotationSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove annotation columns", typeConverter=TypeConverters.toBoolean)
    includeMetadata = Param(Params._dummy(), "includeMetadata", "annotation metadata format", typeConverter=TypeConverters.toBoolean)
    outputAsArray = Param(Params._dummy(), "outputAsArray", "finisher generates an Array with the results instead of string", typeConverter=TypeConverters.toBoolean)
    parseEmbeddingsVectors = Param(Params._dummy(), "parseEmbeddingsVectors", "whether to include embeddings vectors in the process", typeConverter=TypeConverters.toBoolean)

    name = "Finisher"

    @keyword_only
    def __init__(self):
        super(Finisher, self).__init__(classname="com.johnsnowlabs.nlp.Finisher")
        self._setDefault(
            cleanAnnotations=True,
            includeMetadata=False,
            outputAsArray=True,
            parseEmbeddingsVectors=False,
            valueSplitSymbol="#",
            annotationSplitSymbol="@"
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : str
            Input columns for the annotator
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        """Sets column names of finished output annotations.

        Parameters
        ----------
        *value : List[str]
            List of output columns
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

    def setValueSplitSymbol(self, value):
        """Sets character separating values, by default #.

        Parameters
        ----------
        value : str
            Character to separate annotations
        """
        return self._set(valueSplitSymbol=value)

    def setAnnotationSplitSymbol(self, value):
        """Sets character separating annotations, by default @.

        Parameters
        ----------
        value : str
            ...
        """
        return self._set(annotationSplitSymbol=value)

    def setCleanAnnotations(self, value):
        """Sets whether to remove annotation columns, by default True.

        Parameters
        ----------
        value : bool
            Whether to remove annotation columns
        """
        return self._set(cleanAnnotations=value)

    def setIncludeMetadata(self, value):
        """Sets whether to include annotation metadata.

        Parameters
        ----------
        value : bool
            Whether to include annotation metadata
        """
        return self._set(includeMetadata=value)

    def setOutputAsArray(self, value):
        """Sets whether to generate an array with the results instead of a
        string.

        Parameters
        ----------
        value : bool
            Whether to generate an array with the results instead of a string
        """
        return self._set(outputAsArray=value)

    def setParseEmbeddingsVectors(self, value):
        """Sets whether to include embeddings vectors in the process.

        Parameters
        ----------
        value : bool
            Whether to include embeddings vectors in the process
        """
        return self._set(parseEmbeddingsVectors=value)


class EmbeddingsFinisher(AnnotatorTransformer):
    """Extracts embeddings from Annotations into a more easily usable form.

    This is useful for example:

    - WordEmbeddings,
    - Transformer based embeddings such as BertEmbeddings,
    - SentenceEmbeddings and
    - ChunkEmbeddings, etc.

    By using ``EmbeddingsFinisher`` you can easily transform your embeddings
    into array of floats or vectors which are compatible with Spark ML functions
    such as LDA, K-mean, Random Forest classifier or any other functions that
    require a ``featureCol``.

    For more extended examples see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.1_Text_classification_examples_in_SparkML_SparkNLP.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``EMBEDDINGS``         ``NONE``
    ====================== ======================

    Parameters
    ----------
    inputCols
        Names of input annotation columns containing embeddings
    outputCols
        Names of finished output columns
    cleanAnnotations
        Whether to remove all the existing annotation columns, by default False
    outputAsVector
        Whether to output the embeddings as Vectors instead of arrays,
        by default False

    Examples
    --------
    First extract embeddings.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...    .setInputCol("text") \\
    ...    .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...    .setInputCols("document") \\
    ...    .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...    .setInputCols("token") \\
    ...    .setOutputCol("normalized")
    >>> stopwordsCleaner = StopWordsCleaner() \\
    ...    .setInputCols("normalized") \\
    ...    .setOutputCol("cleanTokens") \\
    ...    .setCaseSensitive(False)
    >>> gloveEmbeddings = WordEmbeddingsModel.pretrained() \\
    ...    .setInputCols("document", "cleanTokens") \\
    ...    .setOutputCol("embeddings") \\
    ...    .setCaseSensitive(False)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...    .setInputCols("embeddings") \\
    ...    .setOutputCols("finished_sentence_embeddings") \\
    ...    .setOutputAsVector(True) \\
    ...    .setCleanAnnotations(False)
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library."]]) \\
    ...    .toDF("text")
    >>> pipeline = Pipeline().setStages([
    ...    documentAssembler,
    ...    tokenizer,
    ...    normalizer,
    ...    stopwordsCleaner,
    ...    gloveEmbeddings,
    ...    embeddingsFinisher
    ... ]).fit(data)
    >>> result = pipeline.transform(data)

    Show results.

    >>> resultWithSize = result.selectExpr("explode(finished_sentence_embeddings) as embeddings")
    >>> resultWithSize.show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                      embeddings|
    +--------------------------------------------------------------------------------+
    |[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.685609996318...|
    |[-0.42416998744010925,1.1378999948501587,-0.5717899799346924,-0.5078899860382...|
    |[0.08621499687433243,-0.15772999823093414,-0.06067200005054474,0.395359992980...|
    |[-0.4970499873161316,0.7164199948310852,0.40119001269340515,-0.05761000141501...|
    |[-0.08170200139284134,0.7159299850463867,-0.20677000284194946,0.0295659992843...|
    +--------------------------------------------------------------------------------+

    See Also
    --------
    EmbeddingsFinisher : for finishing embeddings
    """

    inputCols = Param(Params._dummy(), "inputCols", "name of input annotation cols containing embeddings", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output EmbeddingsFinisher ouput cols", typeConverter=TypeConverters.toListString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove all the existing annotation columns", typeConverter=TypeConverters.toBoolean)
    outputAsVector = Param(Params._dummy(), "outputAsVector", "if enabled it will output the embeddings as Vectors instead of arrays", typeConverter=TypeConverters.toBoolean)

    name = "EmbeddingsFinisher"

    @keyword_only
    def __init__(self):
        super(EmbeddingsFinisher, self).__init__(classname="com.johnsnowlabs.nlp.EmbeddingsFinisher")
        self._setDefault(
            cleanAnnotations=False,
            outputAsVector=False
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets name of input annotation columns containing embeddings.

        Parameters
        ----------
        *value : str
            Input columns for the annotator
        """

        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        """Sets names of finished output columns.

        Parameters
        ----------
        *value : List[str]
            Input columns for the annotator
        """

        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

    def setCleanAnnotations(self, value):
        """Sets whether to remove all the existing annotation columns, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to remove all the existing annotation columns
        """

        return self._set(cleanAnnotations=value)

    def setOutputAsVector(self, value):
        """Sets whether to output the embeddings as Vectors instead of arrays,
        by default False.

        Parameters
        ----------
        value : bool
            Whether to output the embeddings as Vectors instead of arrays
        """

        return self._set(outputAsVector=value)


class GraphFinisher(AnnotatorTransformer):
    """Helper class to convert the knowledge graph from GraphExtraction into a
    generic format, such as RDF.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``NONE``
    ====================== ======================

    Parameters
    ----------

    inputCol
        Name of input annotation column
    outputCol
        Name of finisher output column
    cleanAnnotations
        Whether to remove all the existing annotation columns, by default True
    outputAsArray
        Whether to generate an Array with the results, by default True

    Examples
    --------
    This is a continuation of the example of
    :class:`.GraphExtraction`. To see how the graph is extracted, see the
    documentation of that class.

    >>> graphFinisher = GraphFinisher() \\
    ...     .setInputCol("graph") \\
    ...     .setOutputCol("graph_finished")
    ...     .setOutputAsArray(False)
    >>> finishedResult = graphFinisher.transform(result)
    >>> finishedResult.select("text", "graph_finished").show(truncate=False)
    +-----------------------------------------------------+-----------------------------------------------------------------------+
    |text                                                 |graph_finished                                                         |
    +-----------------------------------------------------+-----------------------------------------------------------------------+
    |You and John prefer the morning flight through Denver|[[(prefer,nsubj,morning), (morning,flat,flight), (flight,flat,Denver)]]|
    +-----------------------------------------------------+-----------------------------------------------------------------------+
    """
    inputCol = Param(Params._dummy(), "inputCol", "Name of input annotation col", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "Name of finisher output col", typeConverter=TypeConverters.toString)
    cleanAnnotations = Param(Params._dummy(),
                             "cleanAnnotations",
                             "Whether to remove all the existing annotation columns",
                             typeConverter=TypeConverters.toBoolean)
    outputAsArray = Param(Params._dummy(), "outputAsArray", "Finisher generates an Array with the results",
                          typeConverter=TypeConverters.toBoolean)

    name = "GraphFinisher"

    @keyword_only
    def __init__(self):
        super(GraphFinisher, self).__init__(classname="com.johnsnowlabs.nlp.GraphFinisher")
        self._setDefault(
            cleanAnnotations=True,
            outputAsArray=True
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets name of input annotation column.

        Parameters
        ----------
        value : str
            Name of input annotation column.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """Sets name of finisher output column.

        Parameters
        ----------
        value : str
            Name of finisher output column.
        """
        return self._set(outputCol=value)

    def setCleanAnnotations(self, value):
        """Sets whether to remove all the existing annotation columns, by
        default True.

        Parameters
        ----------
        value : bool
            Whether to remove all the existing annotation columns, by default True.
        """
        return self._set(cleanAnnotations=value)

    def setOutputAsArray(self, value):
        """Sets whether to generate an Array with the results, by default True.

        Parameters
        ----------
        value : bool
            Whether to generate an Array with the results, by default True.
        """
        return self._set(outputAsArray=value)

