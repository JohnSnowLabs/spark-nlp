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
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class MultiDocumentAssembler(AnnotatorTransformer):
    """Prepares data into a format that is processable by Spark NLP.

    This is the entry point for every Spark NLP pipeline. The
    `MultiDocumentAssembler` can read either a ``String`` column or an
    ``Array[String]``. Additionally, :meth:`.setCleanupMode` can be used to
    pre-process the text (Default: ``disabled``). For possible options please
    refer the parameters section.

    For more extended examples on document pre-processing see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-assembler/Loading_Multiple_Documents.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCols: str or List[str]
        Input column name.
    outputCols: str or List[str]
        Output column name.
    idCol: str
        Name of String type column for row id.
    metadataCol: str
        Name of Map type column with metadata information
    cleanupMode: str
        How to cleanup the document , by default disabled.
        Possible values: ``disabled, inplace, inplace_full, shrink, shrink_full,
        each, each_full, delete_full``

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from pyspark.ml import Pipeline
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library."], ["Spark NLP is a state-of-the-art Natural Language Processing library built on top of Apache Spark"]]).toDF("text", "text2")
    >>> documentAssembler = MultiDocumentAssembler().setInputCols(["text", "text2"]).setOutputCols(["document1", "document2"])
    >>> result = documentAssembler.transform(data)
    >>> result.select("document1").show(truncate=False)
    +----------------------------------------------------------------------------------------------+
    |document1                                                                                      |
    +----------------------------------------------------------------------------------------------+
    |[[document, 0, 51, Spark NLP is an open-source text processing library., [sentence -> 0], []]]|
    +----------------------------------------------------------------------------------------------+
    >>> result.select("document1").printSchema()
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

    outputAnnotatorType = AnnotatorType.DOCUMENT

    inputCols = Param(Params._dummy(), "inputCols", "input annotations", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output finished annotation cols", typeConverter=TypeConverters.toListString)
    idCol = Param(Params._dummy(), "idCol", "column for setting an id to such string in row", typeConverter=TypeConverters.toString)
    metadataCol = Param(Params._dummy(), "metadataCol", "String to String map column to use as metadata", typeConverter=TypeConverters.toString)
    cleanupMode = Param(Params._dummy(), "cleanupMode", "possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full", typeConverter=TypeConverters.toString)
    name = 'MultiDocumentAssembler'

    @keyword_only
    def __init__(self):
        super(MultiDocumentAssembler, self).__init__(classname="com.johnsnowlabs.nlp.MultiDocumentAssembler")
        self._setDefault(cleanupMode='disabled')

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : List[str]
            Input columns for the annotator
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        """Sets column names of output annotations.

        Parameters
        ----------
        *value : List[str]
            List of output columns
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

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

    def getOutputCols(self):
        """Gets output columns name of annotations."""
        return self.getOrDefault(self.outputCols)
