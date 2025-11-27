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
"""Contains classes for BertEmbeddings."""

from sparknlp.common import *


class BertEmbeddings(AnnotatorModel,
                     HasEmbeddingsProperties,
                     HasCaseSensitiveProperties,
                     HasStorageRef,
                     HasBatchedAnnotate,
                     HasMaxSentenceLengthLimit):
    """Token-level embeddings using BERT.

    BERT (Bidirectional Encoder Representations from Transformers) provides
    dense vector representations for natural language by using a deep,
    pre-trained neural network with the Transformer architecture.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = BertEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("bert_embeddings")


    The default model is ``"small_bert_L2_768"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_bert.ipynb>`__.
    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch , by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    maxSentenceLength
        Max sentence length to process, by default 128
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
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en") \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("bert_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["bert_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
    |[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
    |[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
    |[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
    |[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
    +--------------------------------------------------------------------------------+
    """

    name = "BertEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

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

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertEmbeddings", java_model=None):
        super(BertEmbeddings, self).__init__(
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
        BertEmbeddings
            The restored model
        """
        from sparknlp.internal import _BertLoader
        jModel = _BertLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return BertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="small_bert_L2_768", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "small_bert_L2_768"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertEmbeddings, name, lang, remote_loc)
