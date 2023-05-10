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
"""Contains classes for RoBertaSentenceEmbeddings."""

from sparknlp.common import *


class RoBertaSentenceEmbeddings(AnnotatorModel,
                                     HasEmbeddingsProperties,
                                     HasCaseSensitiveProperties,
                                     HasStorageRef,
                                     HasBatchedAnnotate,
                                     HasEngine,
                                     HasMaxSentenceLengthLimit):
    """Sentence-level embeddings using RoBERTa. The RoBERTa model was proposed in RoBERTa: A Robustly Optimized BERT
    Pretraining Approach  by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy,
    Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google's BERT model released in 2018. It builds on
    BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much
    larger mini-batches and learning rates. Pretrained models can be loaded with pretrained of the companion object:

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = RoBertaSentenceEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence"]) \\
    ...    .setOutputCol("sentence_embeddings")


    The default model is ``"sent_roberta_base"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

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
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`__

    **Paper abstract:**

    *Language model pretraining has led to significant performance gains but careful comparison between different
    approaches is challenging. Training is computationally expensive, often done on private datasets of different
    sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a
    replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key
    hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed
    the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE,
    RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise
    questions about the source of recently reported improvements. We release our models and code.*

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
    >>> embeddings = RoBertaSentenceEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
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

    name = "RoBertaSentenceEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings", java_model=None):
        super(RoBertaSentenceEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True
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
        BertSentenceEmbeddings
            The restored model
        """
        from sparknlp.internal import _RoBertaSentenceLoader
        jModel = _RoBertaSentenceLoader(folder, spark_session._jsparkSession)._java_obj
        return RoBertaSentenceEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="sent_roberta_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sent_roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        RoBertaSentenceEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(RoBertaSentenceEmbeddings, name, lang, remote_loc)
