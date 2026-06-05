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
"""Contains classes for the BM25 lexical document ranker."""

from sparknlp.common import *
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param


class BM25Approach(AnnotatorApproach):
    """Trains a BM25 (Okapi BM25) lexical ranker over a corpus of tokenized
    documents.

    BM25 is a bag-of-words retrieval function that ranks documents against a
    query based on the query terms appearing in each document. Because a
    document's score depends on corpus-level statistics (how many documents
    contain a term, and the average document length), BM25 is implemented as a
    two-phase Estimator/Model pair:

    - ``BM25Approach`` (this class) scans the full corpus once during ``fit()``
      and learns the document count ``N``, the document frequency ``df(t)`` of
      every term, the average document length ``avgdl`` and the inverse document
      frequency ``idf(t)`` of every term.
    - :class:`BM25Model` reuses those statistics to score every document against
      a user-provided query.

    The input is a column of ``TOKEN`` annotations, so BM25 is normally placed
    after a ``Tokenizer`` (optionally followed by a ``Normalizer`` and/or
    ``StopWordsCleaner``).

    ====================== ================
    Input Annotation types Output Annotation type
    ====================== ================
    ``TOKEN``              ``BM25_RANKINGS``
    ====================== ================

    Parameters
    ----------
    k1
        Term-frequency saturation parameter (typical range [1.0, 2.0]),
        by default 1.2
    b
        Length-normalization parameter (range [0.0, 1.0]), by default 0.75
    minDocFreq
        Drop terms that appear in fewer than this many documents, by default 1
    caseSensitive
        Whether to treat tokens case-sensitively when computing statistics,
        by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document_assembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> stop_words_cleaner = StopWordsCleaner() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("clean_token") \\
    ...     .setCaseSensitive(False)
    >>> bm25 = BM25Approach() \\
    ...     .setInputCols(["clean_token"]) \\
    ...     .setOutputCol("bm25_rankings") \\
    ...     .setK1(1.2) \\
    ...     .setB(0.75) \\
    ...     .setMinDocFreq(1) \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline(stages=[
    ...     document_assembler, tokenizer, stop_words_cleaner, bm25])
    >>> model = pipeline.fit(corpus)
    >>> model.stages[-1].setQuery("vitamin C health benefits fruits")
    >>> model.transform(corpus).selectExpr("explode(bm25_rankings) as r").show()
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.BM25_RANKINGS

    k1 = Param(Params._dummy(),
               "k1",
               "BM25 term-frequency saturation parameter (typical range [1.0, 2.0])",
               typeConverter=TypeConverters.toFloat)

    b = Param(Params._dummy(),
              "b",
              "BM25 length-normalization parameter (range [0.0, 1.0])",
              typeConverter=TypeConverters.toFloat)

    minDocFreq = Param(Params._dummy(),
                       "minDocFreq",
                       "Drop terms that appear in fewer than this many documents",
                       typeConverter=TypeConverters.toInt)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "Whether to treat tokens case-sensitively when computing statistics",
                          typeConverter=TypeConverters.toBoolean)

    def setK1(self, value):
        """Sets the term-frequency saturation parameter k1, by default 1.2.

        Parameters
        ----------
        value : float
            Term-frequency saturation parameter (typical range [1.0, 2.0])
        """
        return self._set(k1=value)

    def setB(self, value):
        """Sets the length-normalization parameter b, by default 0.75.

        Parameters
        ----------
        value : float
            Length-normalization parameter (range [0.0, 1.0])
        """
        return self._set(b=value)

    def setMinDocFreq(self, value):
        """Sets the minimum document frequency for a term to be kept, by default 1.

        Parameters
        ----------
        value : int
            Drop terms that appear in fewer than this many documents
        """
        return self._set(minDocFreq=value)

    def setCaseSensitive(self, value):
        """Sets whether to treat tokens case-sensitively, by default False.

        Parameters
        ----------
        value : bool
            Whether to treat tokens case-sensitively when computing statistics
        """
        return self._set(caseSensitive=value)

    @keyword_only
    def __init__(self):
        super(BM25Approach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.similarity.BM25Approach")
        self._setDefault(
            k1=1.2,
            b=0.75,
            minDocFreq=1,
            caseSensitive=False
        )

    def _create_model(self, java_model):
        return BM25Model(java_model=java_model)


class BM25Model(AnnotatorModel):
    """Fitted model produced by :class:`BM25Approach`.

    It holds the corpus-level statistics (IDF map, average document length and
    document count) and scores every document in a dataset against a query
    using the Okapi BM25 ranking function. The query is provided at transform
    time with ``setQuery(...)``, so the same fitted model can be reused for many
    different queries ("fit once, query many times").

    For every input document the model emits a single ``BM25_RANKINGS``
    annotation whose ``result`` is the BM25 score and whose ``metadata``
    contains ``bm25_score``, ``num_query_terms_matched``, ``query`` and
    ``doc_len``.

    ====================== ================
    Input Annotation types Output Annotation type
    ====================== ================
    ``TOKEN``              ``BM25_RANKINGS``
    ====================== ================

    Parameters
    ----------
    query
        The query to score every document against. Set this at query time.
    k1
        Term-frequency saturation parameter (carried over from the approach)
    b
        Length-normalization parameter (carried over from the approach)
    caseSensitive
        Whether tokens are treated case-sensitively (carried over from the approach)

    Examples
    --------
    >>> from sparknlp.annotator import BM25Model
    >>> loaded = BM25Model.load("/tmp/bm25_corpus_model")
    >>> loaded.setQuery("neural networks deep learning")
    """

    name = "BM25Model"
    inputAnnotatorTypes = [AnnotatorType.TOKEN]
    outputAnnotatorType = AnnotatorType.BM25_RANKINGS

    query = Param(Params._dummy(),
                  "query",
                  "The query to score every document against",
                  typeConverter=TypeConverters.toString)

    k1 = Param(Params._dummy(),
               "k1",
               "BM25 term-frequency saturation parameter (typical range [1.0, 2.0])",
               typeConverter=TypeConverters.toFloat)

    b = Param(Params._dummy(),
              "b",
              "BM25 length-normalization parameter (range [0.0, 1.0])",
              typeConverter=TypeConverters.toFloat)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "Whether to treat tokens case-sensitively when scoring",
                          typeConverter=TypeConverters.toBoolean)

    avgDocLength = Param(Params._dummy(),
                         "avgDocLength",
                         "Average document length (in tokens) of the training corpus",
                         typeConverter=TypeConverters.toFloat)

    numDocuments = Param(Params._dummy(),
                         "numDocuments",
                         "Total number of documents in the training corpus",
                         typeConverter=TypeConverters.toInt)

    def setQuery(self, value):
        """Sets the query that every document is scored against.

        The same fitted model can be re-queried by calling ``setQuery`` again.

        Parameters
        ----------
        value : str
            The query string
        """
        return self._set(query=value)

    def setK1(self, value):
        """Sets the term-frequency saturation parameter k1.

        Parameters
        ----------
        value : float
            Term-frequency saturation parameter (typical range [1.0, 2.0])
        """
        return self._set(k1=value)

    def setB(self, value):
        """Sets the length-normalization parameter b.

        Parameters
        ----------
        value : float
            Length-normalization parameter (range [0.0, 1.0])
        """
        return self._set(b=value)

    def setCaseSensitive(self, value):
        """Sets whether to treat tokens case-sensitively when scoring.

        Parameters
        ----------
        value : bool
            Whether to treat tokens case-sensitively when scoring
        """
        return self._set(caseSensitive=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.similarity.BM25Model",
                 java_model=None):
        super(BM25Model, self).__init__(
            classname=classname,
            java_model=java_model
        )
        # Mirror the Scala-side defaults so the Python params are populated even for a model
        # constructed outside the fit()/load() paths. Learned statistics (idf, avgDocLength,
        # numDocuments) are intentionally left unset; they only come from a fitted model.
        self._setDefault(
            query="",
            k1=1.2,
            b=0.75,
            caseSensitive=False
        )
