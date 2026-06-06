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
    time, so the same fitted model can be reused for many different queries
    ("fit once, query many times"). Provide it either as a raw string with
    ``setQuery(...)`` or — recommended when the corpus was analyzed by a
    non-trivial pipeline — as already-analyzed tokens with
    ``setQueryTokens(...)`` (see the analyzer-symmetry warning below).

    For every input document the model emits a single ``BM25_RANKINGS``
    annotation whose ``result`` is the BM25 score and whose ``metadata``
    contains ``bm25_score``, ``num_query_terms_matched``, ``query`` and
    ``doc_len``.

    .. warning::
        **Analyzer symmetry.** BM25 only scores a query term when it matches a
        key in the learned IDF vocabulary, and those keys were produced by the
        pipeline placed in front of ``BM25Approach`` (``Tokenizer``,
        ``Normalizer``, a stemmer/lemmatizer, ...). A raw-string ``setQuery``
        is only split on non-word characters and lowercased; if your corpus
        pipeline *transforms* tokens (stemming, lemmatization, punctuation
        stripping, ...), a raw query can silently fail to match. In that case
        run the query through the **same** pipeline (e.g. with a
        :class:`LightPipeline`) and pass the resulting tokens to
        ``setQueryTokens``.

    ====================== ================
    Input Annotation types Output Annotation type
    ====================== ================
    ``TOKEN``              ``BM25_RANKINGS``
    ====================== ================

    Parameters
    ----------
    query
        The query to score every document against, as a raw string. The model
        splits it on non-word characters; prefer ``queryTokens`` for non-trivial
        pipelines (see the analyzer-symmetry warning above).
    queryTokens
        The query as a list of already-analyzed terms. When non-empty it
        overrides ``query``. Obtain these by running the query through the same
        pipeline used for the corpus, so query and documents match.
    k1
        Term-frequency saturation parameter (carried over from the approach)
    b
        Length-normalization parameter (carried over from the approach)
    caseSensitive
        Whether tokens are treated case-sensitively. Read-only: it is fixed when
        the corpus statistics are computed and must not be changed on a fitted
        model, so there is no ``setCaseSensitive`` here.

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

    queryTokens = Param(Params._dummy(),
                        "queryTokens",
                        "Pre-analyzed query terms; when non-empty they override the raw query string",
                        typeConverter=TypeConverters.toListString)

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
                          "Whether to treat tokens case-sensitively when scoring (read-only; fixed at fit time)",
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

    def setQueryTokens(self, value):
        """Sets the query as a list of already-analyzed terms.

        When non-empty these override the raw ``query`` string. Obtain them by
        running the query through the same pipeline used for the corpus (for
        example with a :class:`LightPipeline`) so that the query and the
        documents are analyzed identically.

        Parameters
        ----------
        value : List[str]
            Pre-analyzed query terms
        """
        return self._set(queryTokens=value)

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
        """``caseSensitive`` is read-only on a fitted ``BM25Model`` and cannot be set.

        It is fixed when the corpus statistics are computed by :class:`BM25Approach` and is baked
        into the IDF vocabulary keys. Changing it on a fitted model would desynchronize the
        query/document terms from the stored vocabulary and silently corrupt every score, so set
        it on :class:`BM25Approach` before ``fit()`` instead.

        This method is defined only to override the setter that Spark NLP would otherwise generate
        automatically for every parameter; calling it always raises.

        Raises
        ------
        AttributeError
            Always, because ``caseSensitive`` is read-only on the model.
        """
        raise AttributeError(
            "caseSensitive is read-only on a fitted BM25Model: it is fixed when the corpus "
            "statistics are computed by BM25Approach and baked into the IDF vocabulary. Set it on "
            "BM25Approach before fit() instead.")

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
            queryTokens=[],
            k1=1.2,
            b=0.75,
            caseSensitive=False
        )
