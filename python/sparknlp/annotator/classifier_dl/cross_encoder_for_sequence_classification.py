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
"""Contains classes for CrossEncoderForSequenceClassification."""

from sparknlp.common import *


class CrossEncoderForSequenceClassification(AnnotatorModel,
                                            HasCaseSensitiveProperties,
                                            HasBatchedAnnotate,
                                            HasEngine,
                                            HasMaxSentenceLengthLimit):
    """CrossEncoderForSequenceClassification brings cross-encoder scoring (as in
    ``sentence-transformers`` ``CrossEncoder``) into Spark NLP as a first-class annotator.

    It takes two row-aligned document columns, jointly encodes each row's pair as a single
    sequence ``[CLS] text_a [SEP] text_b [SEP]``, runs one forward pass through a BERT-family
    transformer with a classification/regression head, and writes one score per row to a single
    output column. Row ``i`` of the first column and row ``i`` of the second column produce row
    ``i`` of the output — there is no cross-row interaction. Any 1-query-vs-N-candidates reranking
    use case is a ``crossJoin``/``explode`` the user performs upstream.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> crossEncoder = CrossEncoderForSequenceClassification.pretrained() \\
    ...     .setInputCols(["document1", "document2"]) \\
    ...     .setOutputCol("score")

    The default model is ``"cross_encoder_ms_marco_minilm_l6_v2"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Text+Classification>`__.

    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, DOCUMENT`` ``CATEGORY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    maxSentenceLength
        Max sequence length to process. Shared across both texts combined, not
        per text. Defaults to and is capped at the model's
        ``max_position_embeddings``.
    activation
        The activation applied to the logits to obtain the final score. One of
        ``"sigmoid"``, ``"softmax"`` or ``"identity"``, by default ``"sigmoid"``.
    truncationStrategy
        How to truncate a pair when the combined length exceeds
        ``maxSentenceLength``. One of ``"longest_first"`` or ``"query_first"``,
        by default ``"longest_first"``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = MultiDocumentAssembler() \\
    ...     .setInputCols(["query", "passage"]) \\
    ...     .setOutputCols(["document1", "document2"])
    >>> crossEncoder = CrossEncoderForSequenceClassification.pretrained() \\
    ...     .setInputCols(["document1", "document2"]) \\
    ...     .setOutputCol("score")
    >>> pipeline = Pipeline().setStages([document, crossEncoder])
    >>> data = spark.createDataFrame([
    ...     ["How many people live in Berlin?", "Berlin is well known for its museums."]
    ... ]).toDF("query", "passage")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("score.result").show(truncate=False)
    """
    name = "CrossEncoderForSequenceClassification"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CATEGORY

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    activation = Param(Params._dummy(),
                       "activation",
                       "Activation applied to the logits: sigmoid, softmax or identity",
                       TypeConverters.toString)

    truncationStrategy = Param(Params._dummy(),
                               "truncationStrategy",
                               "Pair truncation strategy: longest_first or query_first",
                               TypeConverters.toString)

    modelMaxLength = Param(Params._dummy(),
                           "modelMaxLength",
                           "The model's max sequence length ceiling from its config",
                           TypeConverters.toInt)

    def getClasses(self):
        """Returns labels used to train this model (empty for regression heads)."""
        return self._call_java("getClasses")

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setActivation(self, value):
        """Sets the activation applied to the logits to obtain the final score.

        Parameters
        ----------
        value : str
            One of ``"sigmoid"``, ``"softmax"`` or ``"identity"``
        """
        return self._set(activation=value)

    def setTruncationStrategy(self, value):
        """Sets how a pair is truncated when it exceeds the maximum length.

        Parameters
        ----------
        value : str
            One of ``"longest_first"`` or ``"query_first"``
        """
        return self._set(truncationStrategy=value)

    @keyword_only
    def __init__(self,
                 classname="com.johnsnowlabs.nlp.annotators.classifier.dl.CrossEncoderForSequenceClassification",
                 java_model=None):
        super(CrossEncoderForSequenceClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
            activation="sigmoid",
            truncationStrategy="longest_first"
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        CrossEncoderForSequenceClassification
            The restored model
        """
        from sparknlp.internal import _CrossEncoderForSequenceClassificationLoader
        jModel = _CrossEncoderForSequenceClassificationLoader(folder, spark_session._jsparkSession)._java_obj
        return CrossEncoderForSequenceClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="cross_encoder_ms_marco_minilm_l6_v2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "cross_encoder_ms_marco_minilm_l6_v2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CrossEncoderForSequenceClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CrossEncoderForSequenceClassification, name, lang, remote_loc)
