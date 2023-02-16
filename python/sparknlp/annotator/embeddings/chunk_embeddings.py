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
"""Contains classes for ChunkEmbeddings"""

from sparknlp.common import *


class ChunkEmbeddings(AnnotatorModel):
    """This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate
    chunk embeddings from either Chunker, NGramGenerator, or NerConverter
    outputs.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/embeddings/ChunkEmbeddings.ipynb>`__.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``CHUNK, WORD_EMBEDDINGS`` ``WORD_EMBEDDINGS``
    ========================== ======================

    Parameters
    ----------
    poolingStrategy
        Choose how you would like to aggregate Word Embeddings to Chunk
        Embeddings, by default AVERAGE.
        Possible Values: ``AVERAGE, SUM``
    skipOOV
        Whether to discard default vectors for OOV words from the
        aggregation/pooling.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    Extract the Embeddings from the NGrams

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> nGrams = NGramGenerator() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("chunk") \\
    ...     .setN(2)
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(False)

    Convert the NGram chunks into Word Embeddings

    >>> chunkEmbeddings = ChunkEmbeddings() \\
    ...     .setInputCols(["chunk", "embeddings"]) \\
    ...     .setOutputCol("chunk_embeddings") \\
    ...     .setPoolingStrategy("AVERAGE")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       tokenizer,
    ...       nGrams,
    ...       embeddings,
    ...       chunkEmbeddings
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(chunk_embeddings) as result") \\
    ...     .select("result.annotatorType", "result.result", "result.embeddings") \\
    ...     .show(5, 80)
    +---------------+----------+--------------------------------------------------------------------------------+
    |  annotatorType|    result|                                                                      embeddings|
    +---------------+----------+--------------------------------------------------------------------------------+
    |word_embeddings|   This is|[-0.55661, 0.42829502, 0.86661, -0.409785, 0.06316501, 0.120775, -0.0732005, ...|
    |word_embeddings|      is a|[-0.40674996, 0.22938299, 0.50597, -0.288195, 0.555655, 0.465145, 0.140118, 0...|
    |word_embeddings|a sentence|[0.17417, 0.095253006, -0.0530925, -0.218465, 0.714395, 0.79860497, 0.0129999...|
    |word_embeddings|sentence .|[0.139705, 0.177955, 0.1887775, -0.45545, 0.20030999, 0.461557, -0.07891501, ...|
    +---------------+----------+--------------------------------------------------------------------------------+
    """

    name = "ChunkEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.CHUNK, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    @keyword_only
    def __init__(self):
        super(ChunkEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Chunk Embeddings:" +
                            "AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)
    skipOOV = Param(Params._dummy(), "skipOOV",
                    "Whether to discard default vectors for OOV words from the aggregation / pooling ",
                    typeConverter=TypeConverters.toBoolean)

    def setPoolingStrategy(self, strategy):
        """Sets how to aggregate Word Embeddings to Chunk Embeddings, by default
        AVERAGE.

        Possible Values: ``AVERAGE, SUM``

        Parameters
        ----------
        strategy : str
            Aggregation Strategy
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")

    def setSkipOOV(self, value):
        """Sets whether to discard default vectors for OOV words from the
        aggregation/pooling.

        Parameters
        ----------
        value : bool
            whether to discard default vectors for OOV words from the
            aggregation/pooling.
        """
        return self._set(skipOOV=value)
