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
"""Contains classes for SentenceEmbeddings."""

from sparknlp.common import *


class SentenceEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasStorageRef):
    """Converts the results from WordEmbeddings, BertEmbeddings, or other word
    embeddings into sentence or document embeddings by either summing up or
    averaging all the word embeddings in a sentence or a document (depending on
    the inputCols).

    This can be configured with :meth:`.setPoolingStrategy`, which either be
    ``"AVERAGE"`` or ``"SUM"``.

    For more extended examples see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/Spark_NLP_Spark_ML_Text_Similarity.ipynb>`__..

    ============================= =======================
    Input Annotation types        Output Annotation type
    ============================= =======================
    ``DOCUMENT, WORD_EMBEDDINGS`` ``SENTENCE_EMBEDDINGS``
    ============================= =======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    poolingStrategy
        Choose how you would like to aggregate Word Embeddings to Sentence
        Embeddings: AVERAGE or SUM, by default AVERAGE

    Notes
    -----
    If you choose document as your input for Tokenizer,
    WordEmbeddings/BertEmbeddings, and SentenceEmbeddings then it averages/sums
    all the embeddings into one array of embeddings. However, if you choose
    sentences as inputCols then for each sentence SentenceEmbeddings generates
    one array of embeddings.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsSentence = SentenceEmbeddings() \\
    ...     .setInputCols(["document", "embeddings"]) \\
    ...     .setOutputCol("sentence_embeddings") \\
    ...     .setPoolingStrategy("AVERAGE")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsSentence,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.22093398869037628,0.25130119919776917,0.41810303926467896,-0.380883991718...|
    +--------------------------------------------------------------------------------+
    """

    name = "SentenceEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    @keyword_only
    def __init__(self):
        super(SentenceEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)

    def setPoolingStrategy(self, strategy):
        """Sets how to aggregate the word Embeddings to sentence embeddings, by
        default AVERAGE.

        Can either be AVERAGE or SUM.

        Parameters
        ----------
        strategy : str
            Pooling Strategy, either be AVERAGE or SUM

        Returns
        -------
        [type]
            [description]
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")
