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
"""Contains classes for Doc2Chunk."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.internal import AnnotatorTransformer

from sparknlp.common import AnnotatorProperties, AnnotatorType


class Doc2Chunk(AnnotatorTransformer, AnnotatorProperties):
    """Converts ``DOCUMENT`` type annotations into ``CHUNK`` type with the
    contents of a ``chunkCol``.

    Chunk text must be contained within input ``DOCUMENT``. May be either
    ``StringType`` or ``ArrayType[StringType]`` (using setIsArray). Useful for
    annotators that require a CHUNK type input.

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
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

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

