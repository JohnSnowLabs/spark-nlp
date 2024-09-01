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
"""Contains classes for DocumentSimilarityRanker."""

from sparknlp.common import *
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from sparknlp.internal import AnnotatorTransformer


class DocumentSimilarityRankerApproach(AnnotatorApproach, HasEnableCachingProperties):
    """Annotator that uses LSH techniques present in Spark ML lib to execute
    approximate nearest neighbors search on top of sentence embeddings.

    It aims to capture the semantic meaning of a document in a dense,
    continuous vector space and return it to the ranker search.

    For instantiated/pretrained models, see DocumentSimilarityRankerModel.

    For extended examples of usage, see the jupyter notebook
    `Document Similarity Ranker for Spark NLP <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/doc-sim-ranker/test_doc_sim_ranker.ipynb>`__.

    ======================= ===========================
    Input Annotation types  Output Annotation type
    ======================= ===========================
    ``SENTENCE_EMBEDDINGS`` ``DOC_SIMILARITY_RANKINGS``
    ======================= ===========================

    Parameters
    ----------
    enableCaching
        Whether to enable caching DataFrames or RDDs during the training
    similarityMethod
        The similarity method used to calculate the neighbours.
        (Default: 'brp',Bucketed Random Projection for Euclidean Distance)
    numberOfNeighbours
        The number of neighbours the model will return (Default:`10`)
    bucketLength
        Controls the average size of hash buckets. A larger bucket
        length (i.e., fewer buckets) increases the probability of features
        being hashed to the same bucket (increasing the numbers of true and false positives)
    numHashTables
      Number of hash tables, where increasing number of hash tables lowers the
      false negative rate, and decreasing it improves the running performance.
    visibleDistances
        "Whether to set visibleDistances in ranking output (Default: `false`).
    identityRanking
        Whether to include identity in ranking result set. Useful for debug. (Default: `false`).

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from sparknlp.annotator.similarity.document_similarity_ranker import *
    >>> document_assembler = DocumentAssembler() \
    ...             .setInputCol("text") \
    ...             .setOutputCol("document")
    >>> sentence_embeddings = E5Embeddings.pretrained() \
    ...             .setInputCols(["document"]) \
    ...             .setOutputCol("sentence_embeddings")
    >>> document_similarity_ranker = DocumentSimilarityRankerApproach() \
    ...             .setInputCols("sentence_embeddings") \
    ...             .setOutputCol("doc_similarity_rankings") \
    ...             .setSimilarityMethod("brp") \
    ...             .setNumberOfNeighbours(1) \
    ...             .setBucketLength(2.0) \
    ...             .setNumHashTables(3) \
    ...             .setVisibleDistances(True) \
    ...             .setIdentityRanking(False)
    >>> document_similarity_ranker_finisher = DocumentSimilarityRankerFinisher() \
    ...         .setInputCols("doc_similarity_rankings") \
    ...         .setOutputCols(
    ...             "finished_doc_similarity_rankings_id",
    ...             "finished_doc_similarity_rankings_neighbors") \
    ...         .setExtractNearestNeighbor(True)
    >>> pipeline = Pipeline(stages=[
    ...             document_assembler,
    ...             sentence_embeddings,
    ...             document_similarity_ranker,
    ...             document_similarity_ranker_finisher
    ...         ])
    >>> docSimRankerPipeline = pipeline.fit(data).transform(data)
    >>> (
    ...     docSimRankerPipeline
    ...         .select(
    ...                "finished_doc_similarity_rankings_id",
    ...                "finished_doc_similarity_rankings_neighbors"
    ...         ).show(10, False)
    ... )
    +-----------------------------------+------------------------------------------+
    |finished_doc_similarity_rankings_id|finished_doc_similarity_rankings_neighbors|
    +-----------------------------------+------------------------------------------+
    |1510101612                         |[(1634839239,0.12448559591306324)]        |
    |1634839239                         |[(1510101612,0.12448559591306324)]        |
    |-612640902                         |[(1274183715,0.1220122862046063)]         |
    |1274183715                         |[(-612640902,0.1220122862046063)]         |
    |-1320876223                        |[(1293373212,0.17848855164122393)]        |
    |1293373212                         |[(-1320876223,0.17848855164122393)]       |
    |-1548374770                        |[(-1719102856,0.23297156732534166)]       |
    |-1719102856                        |[(-1548374770,0.23297156732534166)]       |
    +-----------------------------------+------------------------------------------+
    """

    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.DOC_SIMILARITY_RANKINGS

    similarityMethod = Param(Params._dummy(),
                             "similarityMethod",
                             "The similarity method used to calculate the neighbours. (Default: 'brp', "
                             "Bucketed Random Projection for Euclidean Distance)",
                             typeConverter=TypeConverters.toString)

    numberOfNeighbours = Param(Params._dummy(),
                               "numberOfNeighbours",
                               "The number of neighbours the model will return (Default:`10`)",
                               typeConverter=TypeConverters.toInt)

    bucketLength = Param(Params._dummy(),
                         "bucketLength",
                         "The bucket length that controls the average size of hash buckets. "
                         "A larger bucket length (i.e., fewer buckets) increases the probability of features "
                         "being hashed to the same bucket (increasing the numbers of true and false positives).",
                         typeConverter=TypeConverters.toFloat)

    numHashTables = Param(Params._dummy(),
                          "numHashTables",
                          "number of hash tables, where increasing number of hash tables lowers the "
                          "false negative rate,and decreasing it improves the running performance.",
                          typeConverter=TypeConverters.toInt)

    visibleDistances = Param(Params._dummy(),
                             "visibleDistances",
                             "Whether to set visibleDistances in ranking output (Default: `false`).",
                             typeConverter=TypeConverters.toBoolean)

    identityRanking = Param(Params._dummy(),
                            "identityRanking",
                            "Whether to include identity in ranking result set. Useful for debug. (Default: `false`).",
                            typeConverter=TypeConverters.toBoolean)

    asRetrieverQuery = Param(Params._dummy(),
                             "asRetrieverQuery",
                             "Whether to set the model as retriever RAG with a specific query string."
                             "(Default: `empty`)",
                             typeConverter=TypeConverters.toString)

    aggregationMethod = Param(Params._dummy(),
                              "aggregationMethod",
                              "Specifies the method used to aggregate multiple sentence embeddings into a single vector representation.",
                              typeConverter=TypeConverters.toString)


    def setSimilarityMethod(self, value):
        """Sets the similarity method used to calculate the neighbours.
            (Default: `"brp"`, Bucketed Random Projection for Euclidean Distance)

        Parameters
        ----------
        value : str
            the similarity method to calculate the neighbours.
        """
        return self._set(similarityMethod=value)

    def setNumberOfNeighbours(self, value):
        """Sets The number of neighbours the model will return for each document(Default:`"10"`).

        Parameters
        ----------
        value : str
            the number of neighbours the model will return for each document.
        """
        return self._set(numberOfNeighbours=value)

    def setBucketLength(self, value):
        """Sets the bucket length that controls the average size of hash buckets (Default:`"2.0"`).

        Parameters
        ----------
        value : float
            Sets the bucket length that controls the average size of hash buckets.
        """
        return self._set(bucketLength=value)

    def setNumHashTables(self, value):
        """Sets the number of hash tables.

        Parameters
        ----------
        value : int
            Sets the number of hash tables.
        """
        return self._set(numHashTables=value)

    def setVisibleDistances(self, value):
        """Sets the document distances visible in the result set.

        Parameters
        ----------
        value : bool
            Sets the document distances visible in the result set.
            Default('False')
        """
        return self._set(visibleDistances=value)

    def setIdentityRanking(self, value):
        """Sets the document identity ranking inclusive in the result set.

        Parameters
        ----------
        value : bool
            Sets the document identity ranking inclusive in the result set.
            Useful for debugging.
            Default('False').
        """
        return self._set(identityRanking=value)

    def asRetriever(self, value):
        """Sets the query to use the document similarity ranker as a retriever in a RAG fashion.
            (Default: `""`, empty if this annotator is not used as retriever)

        Parameters
        ----------
        value : str
             the query to use to select nearest neighbors in the retrieval process.
        """
        return self._set(asRetrieverQuery=value)

    def setAggregationMethod(self, value):
        """Set the method used to aggregate multiple sentence embeddings into a single vector
           representation.

        Parameters
        ----------
        value : str
           Options include
            'AVERAGE' (compute the mean of all embeddings),
            'FIRST' (use the first embedding only),
            'MAX' (compute the element-wise maximum across embeddings)

           Default ('AVERAGE')
        """
        return self._set(aggregationMethod=value)

    @keyword_only
    def __init__(self):
        super(DocumentSimilarityRankerApproach, self)\
            .__init__(classname="com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach")
        self._setDefault(
            similarityMethod="brp",
            numberOfNeighbours=10,
            bucketLength=2.0,
            numHashTables=3,
            visibleDistances=False,
            identityRanking=False,
            asRetrieverQuery=""
        )

    def _create_model(self, java_model):
        return DocumentSimilarityRankerModel(java_model=java_model)


class DocumentSimilarityRankerModel(AnnotatorModel, HasEmbeddingsProperties):

    name = "DocumentSimilarityRankerModel"
    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]
    outputAnnotatorType = AnnotatorType.DOC_SIMILARITY_RANKINGS

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerModel",
                 java_model=None):
        super(DocumentSimilarityRankerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class DocumentSimilarityRankerFinisher(AnnotatorTransformer):
    """Instantiated model of the DocumentSimilarityRankerApproach. For usage and examples see the
    documentation of the main class.

    ======================= ===========================
    Input Annotation types  Output Annotation type
    ======================= ===========================
    ``SENTENCE_EMBEDDINGS`` ``DOC_SIMILARITY_RANKINGS``
    ======================= ===========================

    Parameters
    ----------
    extractNearestNeighbor
        Whether to extract the nearest neighbor document
    """
    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "name of input annotation cols containing document similarity ranker results",
                      typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(),
                       "outputCols",
                       "output DocumentSimilarityRankerFinisher output cols",
                       typeConverter=TypeConverters.toListString)
    extractNearestNeighbor = Param(Params._dummy(), "extractNearestNeighbor",
                             "whether to extract the nearest neighbor document",
                             typeConverter=TypeConverters.toBoolean)

    name = "DocumentSimilarityRankerFinisher"

    @keyword_only
    def __init__(self):
        super(DocumentSimilarityRankerFinisher, self).__init__(classname="com.johnsnowlabs.nlp.finisher.DocumentSimilarityRankerFinisher")
        self._setDefault(
            extractNearestNeighbor=False
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

    def setExtractNearestNeighbor(self, value):
        """Sets whether to extract the nearest neighbor document, by default False.

        Parameters
        ----------
        value : bool
            Whether to extract the nearest neighbor document
        """

        return self._set(extractNearestNeighbor=value)

    def getInputCols(self):
        """Gets input columns name of annotations."""
        return self.getOrDefault(self.inputCols)

    def getOutputCols(self):
        """Gets output columns name of annotations."""
        if len(self.getOrDefault(self.outputCols)) == 0:
            return ["finished_" + input_col for input_col in self.getInputCols()]
        else:
            return self.getOrDefault(self.outputCols)