#  Copyright 2017-2025 John Snow Labs
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
"""Contains class for PairwiseVectorSimilarity."""

from sparknlp.common import *
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param


class PairwiseVectorSimilarity(AnnotatorModel):
    """Computes pairwise vector similarity between two sets of sentence embeddings.

    Takes **two** ``SENTENCE_EMBEDDINGS`` input columns (e.g. query embeddings and document
    embeddings already joined on the same row) and scores all N×M pairs between the embeddings
    in column A and the embeddings in column B. Each pair produces one ``VECTOR_SIMILARITY``
    output annotation whose ``result`` holds the score as a string, and whose ``metadata``
    contains typed fields for easy extraction.

    **Sign conventions**

    ============  ===========  ========================
    Method        Range        Higher means
    ============  ===========  ========================
    cosine        [-1.0, 1.0]  more similar
    dotProduct    (−∞, +∞)     more similar
    euclidean     (−∞, 0.0]    more similar (0.0 = identical vectors)
    ============  ===========  ========================

    The ``euclidean`` method returns the **negative** L2 distance so that "higher is better"
    holds uniformly across all three methods. A score of ``0.0`` means the two vectors are
    identical.

    **Important: input data shape**

    Each input column should contain **exactly one** embedding per row for standard document
    retrieval. If a column contains N > 1 embeddings (e.g. from ``SentenceDetector`` + embedder),
    all N×M cross-pairs are scored and returned as separate annotations.

    To compare queries against a corpus, crossJoin them first so each row holds one
    (query, document) pair:

    ======================= ========================
    Input Annotation types  Output Annotation type
    ======================= ========================
    ``SENTENCE_EMBEDDINGS`` ``VECTOR_SIMILARITY``
    ``SENTENCE_EMBEDDINGS``
    ======================= ========================

    Parameters
    ----------
    similarityMethod
        Similarity function: ``"cosine"`` (default), ``"dotProduct"``, or ``"euclidean"``
        (negative L2 distance).

    Examples
    --------
    >>> from sparknlp.annotator import PairwiseVectorSimilarity
    >>> from pyspark.sql.functions import col, explode, desc

    >>> # Assume embedding_pipeline produces a "embeddings" SENTENCE_EMBEDDINGS column.
    >>> query_df = embedding_pipeline.transform(queries).select(col("embeddings").alias("query_emb"))
    >>> corpus_df = embedding_pipeline.transform(corpus).select(col("embeddings").alias("doc_emb"), col("id"))
    >>> paired = query_df.crossJoin(corpus_df)

    >>> pvs = PairwiseVectorSimilarity() \\
    ...     .setInputCols(["query_emb", "doc_emb"]) \\
    ...     .setOutputCol("similarity") \\
    ...     .setSimilarityMethod("cosine")

    >>> result = pvs.transform(paired) \\
    ...     .select(explode(col("similarity")).alias("s")) \\
    ...     .select(
    ...         col("s.metadata")["sentence_a_text"].alias("query"),
    ...         col("s.metadata")["sentence_b_text"].alias("document"),
    ...         col("s.result").cast("double").alias("score")) \\
    ...     .orderBy(desc("score"))
    >>> result.show(truncate=False)
    """

    name = "PairwiseVectorSimilarity"
    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS, AnnotatorType.SENTENCE_EMBEDDINGS]
    outputAnnotatorType = AnnotatorType.VECTOR_SIMILARITY

    similarityMethod = Param(
        Params._dummy(),
        "similarityMethod",
        'Similarity function: "cosine" (default), "dotProduct", or "euclidean" '
        '(negative L2 distance — 0.0 = identical, more negative = less similar).',
        typeConverter=TypeConverters.toString)

    def setSimilarityMethod(self, value):
        """Sets the similarity function used to score pairs.

        Parameters
        ----------
        value : str
            One of ``"cosine"`` (default), ``"dotProduct"``, or ``"euclidean"``.
            The ``"euclidean"`` option returns the **negative** L2 distance so that
            higher scores consistently mean more similar across all three methods.
        """
        return self._set(similarityMethod=value)

    def getSimilarityMethod(self):
        """Gets the currently configured similarity method.

        Returns
        -------
        str
            The similarity method name.
        """
        return self.getOrDefault(self.similarityMethod)

    @keyword_only
    def __init__(self,
                 classname="com.johnsnowlabs.nlp.annotators.similarity.PairwiseVectorSimilarity",
                 java_model=None):
        super(PairwiseVectorSimilarity, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(similarityMethod="cosine")
