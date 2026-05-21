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
"""Contains classes for LateChunkEmbeddings."""

from sparknlp.common import *

__all__ = ["LateChunkEmbeddings"]


class LateChunkEmbeddings(AnnotatorModel):
    """Produces contextual chunk-level embeddings using the **Late Chunking** technique
    described in `Jin et al. (2024) <https://arxiv.org/abs/2409.04701>`__.

    Unlike :class:`.ChunkEmbeddings`, which embeds each chunk in isolation,
    ``LateChunkEmbeddings`` expects that the upstream token-embedding stage (e.g.
    :class:`.ModernBertEmbeddings` or :class:`.LongformerEmbeddings`) has already
    processed the **full document** in a single forward pass, producing contextual
    token representations. This annotator then locates the tokens that fall within
    each chunk's character span and mean-pools them into a single
    ``SENTENCE_EMBEDDINGS`` vector — so every chunk embedding is informed by the
    complete document context rather than being isolated.

    .. note::
        ``LateChunkEmbeddings`` **must** appear **after** the token-embedding stage
        in the pipeline. Placing it before the embedding stage will raise a runtime
        error.

    .. note::
        The contextual benefit is bounded by the upstream model's maximum sequence
        length (e.g. 8 192 tokens for ``ModernBertEmbeddings``). Documents that
        exceed this limit are truncated before embedding, which reduces cross-chunk
        context for tokens near the end of very long documents.

    ====================================== ======================
    Input Annotation types                 Output Annotation type
    ====================================== ======================
    ``DOCUMENT, CHUNK, WORD_EMBEDDINGS``   ``SENTENCE_EMBEDDINGS``
    ====================================== ======================

    Parameters
    ----------
    poolingStrategy
        Strategy to aggregate token embeddings within each chunk span, by default
        ``AVERAGE``.
        Possible values: ``AVERAGE``, ``SUM``
    skipOOV
        Whether to discard default zero-vectors for OOV tokens from the pool,
        by default ``True``.

    References
    ----------
    Jin et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context
    Embedding Models*, arXiv:2409.04701 (2024).

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    Build a late-chunking retrieval pipeline

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> tokenEmbeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("token_embeddings") \\
    ...     .setMaxSentenceLength(8192)
    >>> chunker = Doc2Chunk() \\
    ...     .setInputCols(["document"]) \\
    ...     .setChunkCol("chunks") \\
    ...     .setIsArray(True) \\
    ...     .setOutputCol("chunk")
    >>> lateChunkEmbeddings = LateChunkEmbeddings() \\
    ...     .setInputCols(["document", "chunk", "token_embeddings"]) \\
    ...     .setOutputCol("late_chunk_embeddings") \\
    ...     .setPoolingStrategy("AVERAGE")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       tokenEmbeddings,
    ...       chunker,
    ...       lateChunkEmbeddings
    ...     ])
    >>> data = spark.createDataFrame([(
    ...     "AcmeDrug was prescribed for migraine in March. The patient took two doses.\\n\\n"
    ...     "It caused severe nausea the next day, and therapy was stopped.",
    ...     [
    ...         "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
    ...         "It caused severe nausea the next day, and therapy was stopped."
    ...     ]
    ... )], ["text", "chunks"])
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(late_chunk_embeddings) as r") \\
    ...     .select("r.annotatorType", "r.result", "r.embeddings") \\
    ...     .show(5, 80)
    +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
    |      annotatorType|                                                                    result|                                                                      embeddings|
    +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
    |sentence_embeddings|AcmeDrug was prescribed for migraine in March. The patient took two doses.|[0.050471008, -0.07595207, 0.031268876, 0.15105441, -0.013697156, 0.08131724,...|
    |sentence_embeddings|            It caused severe nausea the next day, and therapy was stopped.|[0.0735685, 0.0060829176, 0.12051964, 0.22399232, 0.055884164, 0.066795066, 0...|
    +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
    """

    name = "LateChunkEmbeddings"

    inputAnnotatorTypes = [
        AnnotatorType.DOCUMENT,
        AnnotatorType.CHUNK,
        AnnotatorType.WORD_EMBEDDINGS,
    ]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    @keyword_only
    def __init__(self):
        super(LateChunkEmbeddings, self).__init__(
            classname="com.johnsnowlabs.nlp.embeddings.LateChunkEmbeddings"
        )
        self._setDefault(poolingStrategy="AVERAGE", skipOOV=True)

    poolingStrategy = Param(
        Params._dummy(),
        "poolingStrategy",
        "Strategy to aggregate token embeddings into a chunk embedding: AVERAGE or SUM",
        typeConverter=TypeConverters.toString,
    )

    skipOOV = Param(
        Params._dummy(),
        "skipOOV",
        "Whether to discard default vectors for OOV words from the aggregation / pooling",
        typeConverter=TypeConverters.toBoolean,
    )

    def setPoolingStrategy(self, strategy):
        """Sets the strategy used to aggregate token embeddings within each chunk span.

        Parameters
        ----------
        strategy : str
            Pooling strategy. One of ``AVERAGE`` (default) or ``SUM``.
        """
        if strategy in ("AVERAGE", "SUM"):
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")

    def setSkipOOV(self, value):
        """Sets whether to discard default zero-vectors for OOV tokens during pooling.

        Parameters
        ----------
        value : bool
            If ``True`` (default), OOV zero-vectors are excluded from the pool so that
            they do not dilute the chunk embedding.
        """
        return self._set(skipOOV=value)

