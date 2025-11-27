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
"""Contains classes for BertSentenceEmbeddings."""

from sparknlp.common import *


class BertSentenceEmbeddings(AnnotatorModel,
                                  HasEmbeddingsProperties,
                                  HasCaseSensitiveProperties,
                                  HasStorageRef,
                                  HasBatchedAnnotate,
                                  HasEngine,
                                  HasMaxSentenceLengthLimit):
    """Sentence-level embeddings using BERT. BERT (Bidirectional Encoder
    Representations from Transformers) provides dense vector representations for
    natural language by using a deep, pre-trained neural network with the
    Transformer architecture.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>>embeddings = BertSentenceEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence"]) \\
    ...    .setOutputCol("sentence_bert_embeddings")


    The default model is ``"sent_small_bert_L2_768"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb>`__.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``DOCUMENT``           ``SENTENCE_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    dimension
        Number of embedding dimensions, by default 768
    maxSentenceLength
        Max sentence length to process, by default 128
    isLong
        Use Long type instead of Int type for inputs buffer - Some Bert models
        require Long instead of Int.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__

    https://github.com/google-research/bert

    **Paper abstract**

    *We introduce a new language representation model called BERT, which stands
    for Bidirectional Encoder Representations from Transformers. Unlike recent
    language representation models, BERT is designed to pre-train deep
    bidirectional representations from unlabeled text by jointly conditioning on
    both left and right context in all layers. As a result, the pre-trained BERT
    model can be fine-tuned with just one additional output layer to create
    state-of-the-art models for a wide range of tasks, such as question
    answering and language inference, without substantial task-specific
    architecture modifications. BERT is conceptually simple and empirically
    powerful. It obtains new state-of-the-art results on eleven natural language
    processing tasks, including pushing the GLUE score to 80.5% (7.7% point
    absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute
    improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point
    absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
    improvement).*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_bert_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_bert_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["John loves apples. Mary loves oranges. John loves Mary."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.8951074481010437,0.13753940165042877,0.3108254075050354,-1.65693199634552...|
    |[-0.6180210709571838,-0.12179657071828842,-0.191165953874588,-1.4497021436691...|
    |[-0.822715163230896,0.7568016648292542,-0.1165061742067337,-1.59048593044281,...|
    +--------------------------------------------------------------------------------+
    """

    name = "BertSentenceEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    isLong = Param(Params._dummy(),
                   "isLong",
                   "Use Long type instead of Int type for inputs buffer - Some Bert models require Long instead of Int.",
                   typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setIsLong(self, value):
        """Sets whether to use Long type instead of Int type for inputs buffer.

        Some Bert models require Long instead of Int.

        Parameters
        ----------
        value : bool
            Whether to use Long type instead of Int type for inputs buffer
        """
        return self._set(isLong=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings", java_model=None):
        super(BertSentenceEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession
        use_openvino: bool
            Use OpenVINO backend

        Returns
        -------
        BertSentenceEmbeddings
            The restored model
        """
        from sparknlp.internal import _BertSentenceLoader
        jModel = _BertSentenceLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return BertSentenceEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="sent_small_bert_L2_768", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sent_small_bert_L2_768"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertSentenceEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertSentenceEmbeddings, name, lang, remote_loc)
