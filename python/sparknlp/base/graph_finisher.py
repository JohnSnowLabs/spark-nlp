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
"""Contains classes for the GraphFinisher."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from sparknlp.internal import AnnotatorTransformer


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

