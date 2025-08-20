#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for the GGUFRankingFinisher."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from sparknlp.internal import AnnotatorTransformer


class GGUFRankingFinisher(AnnotatorTransformer):
    """Finisher for AutoGGUFReranker outputs that provides ranking capabilities
    including top-k selection, sorting by relevance score, and score normalization.

    This finisher processes the output of AutoGGUFReranker, which contains documents with
    relevance scores in their metadata. It provides several options for post-processing:

    - Top-k selection: Select only the top k documents by relevance score
    - Score thresholding: Filter documents by minimum relevance score
    - Min-max scaling: Normalize relevance scores to 0-1 range
    - Sorting: Sort documents by relevance score in descending order
    - Ranking: Add rank information to document metadata

    The finisher preserves the document annotation structure while adding ranking information
    to the metadata and optionally filtering/sorting the documents.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/finisher/gguf_ranking_finisher_example.py>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCols
        Name of input annotation columns containing reranked documents
    outputCol
        Name of output annotation column containing ranked documents, by default "ranked_documents"
    topK
        Maximum number of top documents to return based on relevance score (-1 for no limit), by default -1
    minRelevanceScore
        Minimum relevance score threshold for filtering documents, by default Double.MinValue
    minMaxScaling
        Whether to apply min-max scaling to normalize relevance scores to 0-1 range, by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> reranker = AutoGGUFReranker.pretrained("bge-reranker-v2-m3-Q4_K_M") \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("reranked_documents") \\
    ...     .setQuery("A man is eating pasta.")
    >>> finisher = GGUFRankingFinisher() \\
    ...     .setInputCols("reranked_documents") \\
    ...     .setOutputCol("ranked_documents") \\
    ...     .setTopK(3) \\
    ...     .setMinMaxScaling(True)
    >>> pipeline = Pipeline().setStages([documentAssembler, reranker, finisher])
    >>> data = spark.createDataFrame([
    ...     ("A man is eating food.",),
    ...     ("A man is eating a piece of bread.",),
    ...     ("The girl is carrying a baby.",),
    ...     ("A man is riding a horse.",)
    ... ], ["text"])
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("ranked_documents").show(truncate=False)
    # Documents will be sorted by relevance with rank information in metadata
    """

    name = "GGUFRankingFinisher"

    inputCols = Param(Params._dummy(),
                     "inputCols",
                     "Name of input annotation columns containing reranked documents",
                     typeConverter=TypeConverters.toListString)

    outputCol = Param(Params._dummy(),
                     "outputCol", 
                     "Name of output annotation column containing ranked documents",
                     typeConverter=TypeConverters.toListString)

    topK = Param(Params._dummy(),
                 "topK",
                 "Maximum number of top documents to return based on relevance score (-1 for no limit)",
                 typeConverter=TypeConverters.toInt)

    minRelevanceScore = Param(Params._dummy(),
                             "minRelevanceScore",
                             "Minimum relevance score threshold for filtering documents",
                             typeConverter=TypeConverters.toFloat)

    minMaxScaling = Param(Params._dummy(),
                         "minMaxScaling",
                         "Whether to apply min-max scaling to normalize relevance scores to 0-1 range",
                         typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(GGUFRankingFinisher, self).__init__(
            classname="com.johnsnowlabs.nlp.finisher.GGUFRankingFinisher")
        self._setDefault(
            topK=-1,
            minRelevanceScore=float('-inf'),  # Equivalent to Double.MinValue
            minMaxScaling=False,
            outputCol=["ranked_documents"]
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets input annotation column names.

        Parameters
        ----------
        value : List[str]
            Input annotation column names containing reranked documents
        """
        if len(value) == 1 and isinstance(value[0], list):
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def getInputCols(self):
        """Gets input annotation column names.

        Returns
        -------
        List[str]
            Input annotation column names
        """
        return self.getOrDefault(self.inputCols)

    def setOutputCol(self, value):
        """Sets output annotation column name.

        Parameters
        ----------
        value : str
            Output annotation column name
        """
        return self._set(outputCol=[value])

    def getOutputCol(self):
        """Gets output annotation column name.

        Returns
        -------
        str
            Output annotation column name
        """
        output_cols = self.getOrDefault(self.outputCol)
        return output_cols[0] if output_cols else "ranked_documents"

    def setTopK(self, value):
        """Sets maximum number of top documents to return.

        Parameters
        ----------
        value : int
            Maximum number of top documents to return (-1 for no limit)
        """
        return self._set(topK=value)

    def getTopK(self):
        """Gets maximum number of top documents to return.

        Returns
        -------
        int
            Maximum number of top documents to return
        """
        return self.getOrDefault(self.topK)

    def setMinRelevanceScore(self, value):
        """Sets minimum relevance score threshold.

        Parameters
        ----------
        value : float
            Minimum relevance score threshold
        """
        return self._set(minRelevanceScore=value)

    def getMinRelevanceScore(self):
        """Gets minimum relevance score threshold.

        Returns
        -------
        float
            Minimum relevance score threshold
        """
        return self.getOrDefault(self.minRelevanceScore)

    def setMinMaxScaling(self, value):
        """Sets whether to apply min-max scaling.

        Parameters
        ----------
        value : bool
            Whether to apply min-max scaling to normalize scores
        """
        return self._set(minMaxScaling=value)

    def getMinMaxScaling(self):
        """Gets whether to apply min-max scaling.

        Returns
        -------
        bool
            Whether min-max scaling is enabled
        """
        return self.getOrDefault(self.minMaxScaling)
