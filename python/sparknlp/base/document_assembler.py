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
"""Contains classes for the DocumentAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class DocumentAssembler(AnnotatorTransformer):
    """Prepares data into a format that is processable by Spark NLP.

    This is the entry point for every Spark NLP pipeline. The
    `DocumentAssembler` reads ``String`` columns. Additionally,
    :meth:`.setCleanupMode` can be used to pre-process the
    text (Default: ``disabled``). For possible options please refer the
    parameters section.

    For more extended examples on document pre-processing see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-assembler/Loading_Documents_With_DocumentAssembler.ipynb>`__.

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

    outputAnnotatorType = AnnotatorType.DOCUMENT

    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    idCol = Param(Params._dummy(), "idCol", "column for setting an id to such string in row", typeConverter=TypeConverters.toString)
    metadataCol = Param(Params._dummy(), "metadataCol", "String to String map column to use as metadata", typeConverter=TypeConverters.toString)
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

    def getOutputCol(self):
        """Gets output column name of annotations."""
        return self.getOrDefault(self.outputCol)

    # def getInputCol(self):
    #     """Gets current column names of input annotations."""
    #     return self.getOrDefault(self.inputCol)
