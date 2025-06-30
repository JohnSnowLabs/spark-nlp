#  Copyright 2017-2023 John Snow Labs
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
"""Contains classes for Date2Chunk."""

from sparknlp.common import *


class Date2Chunk(AnnotatorModel):
    """Converts ``DATE`` type Annotations to ``CHUNK`` type.

    This can be useful if the following annotators after DateMatcher and MultiDateMatcher require ```CHUNK``` types.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DATE``               ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    entityName
        Entity name for the metadata, by default ``"DATE"``.

    Examples
    --------
    >>> from pyspark.ml import Pipeline

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> date = DateMatcher() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("date")
    >>> date2Chunk = Date2Chunk() \\
    ...     .setInputCols(["date"]) \\
    ...     .setOutputCol("date_chunk")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     date,
    ...     date2Chunk
    ... ])
    >>> data = spark.createDataFrame([["Omicron is a new variant of COVID-19, which the World Health Organization designated a variant of concern on Nov. 26, 2021/26/11."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("date_chunk").show(1, truncate=False)
       ----------------------------------------------------+
       |date_chunk                                          |
       ----------------------------------------------------+
       |[{chunk, 118, 121, 2021/01/01, {sentence -> 0}, []}]|
       ----------------------------------------------------+
    """
    name = "Date2Chunk"

    inputAnnotatorTypes = [AnnotatorType.DATE]

    outputAnnotatorType = AnnotatorType.CHUNK

    @keyword_only
    def __init__(self):
        super(Date2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Date2Chunk")
        self._setDefault(entityName="DATE")

    entityName = Param(Params._dummy(), "entityName", "Entity name for the metadata",
                       TypeConverters.toString)

    def setEntityName(self, name):
        """Sets Learning Rate, by default 0.001.

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(entityName=name)
        return self
