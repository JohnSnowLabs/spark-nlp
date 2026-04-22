#  Copyright 2017-2026 John Snow Labs
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
"""Contains classes for the MultiColumnAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.internal import AnnotatorTransformer
from sparknlp.common import AnnotatorProperties, AnnotatorType


class MultiColumnAssembler(AnnotatorTransformer, AnnotatorProperties):
    """Merges multiple annotation columns into a single annotation column.

    This is useful when multiple annotators produce separate annotation columns
    (e.g., ``document_text``, ``document_table`` from ``ReaderAssembler``) and a
    downstream annotator (e.g., ``AutoGGUFVisionModel``) expects a single input
    column containing all annotations.

    Annotations from all input columns are collected and concatenated into the
    output column. The output annotator type defaults to ``DOCUMENT`` but can be
    configured. Each annotation's metadata is preserved, and a ``source_column``
    key is added to track the original column name.

    **Note:** All input columns must use the ``Annotation`` schema. Columns
    using ``AnnotationImage`` schema (e.g., IMAGE-typed columns from
    ``ReaderAssembler``) are not supported.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCols
        Input annotation columns to merge
    outputCol
        Output annotation column name
    outputAsAnnotatorType
        The annotator type to use for the output annotations (Default: ``document``)
    sortByBegin
        Whether to sort merged annotations by their begin position (Default: ``False``)

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler1 = DocumentAssembler() \\
    ...     .setInputCol("text1") \\
    ...     .setOutputCol("document_text")
    >>> documentAssembler2 = DocumentAssembler() \\
    ...     .setInputCol("text2") \\
    ...     .setOutputCol("document_table")
    >>> multiColumnAssembler = MultiColumnAssembler() \\
    ...     .setInputCols(["document_text", "document_table"]) \\
    ...     .setOutputCol("merged_document")
    >>> data = spark.createDataFrame([("Hello world", "Name | Age")]).toDF("text1", "text2")
    >>> pipeline = Pipeline().setStages([documentAssembler1, documentAssembler2, multiColumnAssembler]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.selectExpr("merged_document.result").show(truncate=False)
    +---------------------------+
    |result                     |
    +---------------------------+
    |[Hello world, Name | Age] |
    +---------------------------+
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    outputAsAnnotatorType = Param(
        Params._dummy(),
        "outputAsAnnotatorType",
        "The annotator type to use for the output annotations (Default: document)",
        typeConverter=TypeConverters.toString,
    )

    sortByBegin = Param(
        Params._dummy(),
        "sortByBegin",
        "Whether to sort merged annotations by their begin position (Default: False)",
        typeConverter=TypeConverters.toBoolean,
    )

    name = "MultiColumnAssembler"

    @keyword_only
    def __init__(self):
        super(MultiColumnAssembler, self).__init__(
            classname="com.johnsnowlabs.nlp.MultiColumnAssembler"
        )
        self._setDefault(outputAsAnnotatorType="document", sortByBegin=False)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets input annotation columns to merge.

        Parameters
        ----------
        *value : str
            Input column names
        """
        if type(value[0]) == str or type(value[0]) == list:
            if len(value) == 1 and type(value[0]) == list:
                return self._set(inputCols=value[0])
            else:
                return self._set(inputCols=list(value))
        else:
            raise TypeError("InputCols datatype not supported. It must be either str or list")

    def setOutputAsAnnotatorType(self, value):
        """Sets the annotator type for the output annotations.

        Parameters
        ----------
        value : str
            The annotator type (e.g., "document", "chunk", "table")
        """
        return self._set(outputAsAnnotatorType=value)

    def setSortByBegin(self, value):
        """Sets whether to sort merged annotations by begin position.

        Parameters
        ----------
        value : bool
            Whether to sort by begin position
        """
        return self._set(sortByBegin=value)

