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
"""Contains classes for the Finisher."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from sparknlp.internal import AnnotatorTransformer


class Finisher(AnnotatorTransformer):
    """Converts annotation results into a format that easier to use.

    It is useful to extract the results from Spark NLP Pipelines. The Finisher
    outputs annotation(s) values into ``String``.

    For more extended examples on document pre-processing see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/model-downloader/Create%20custom%20pipeline%20-%20NerDL.ipynb
>`__.

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
            annotationSplitSymbol="@",
            outputCols=[]
        )

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

    def getInputCols(self):
        """Gets input columns name of annotations."""
        return self.getOrDefault(self.inputCols)

    def getOutputCols(self):
        """Gets output columns name of annotations."""
        if len(self.getOrDefault(self.outputCols)) == 0:
            return ["finished_" + input_col for input_col in self.getInputCols()]
        else:
            return self.getOrDefault(self.outputCols)
