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
"""Contains classes for E5Embeddings."""

from sparknlp.common import *


class MPNetEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasMaxSentenceLengthLimit):
    """Sentence embeddings using MPNet.

    MPNet adopts a novel pre-training method, named masked and permuted language modeling,
    to inherit the advantages of masked language modeling and permuted language modeling for
    natural language understanding.

    Note that this annotator is only supported for Spark Versions 3.4 and up.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = MPNetEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("mpnet_embeddings")


    The default model is ``"all_mpnet_base_v2"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=MPNet>`__.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
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
        Max sentence length to process, by default 512
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `MPNet: Masked and Permuted Pre-training for Language Understanding <https://arxiv.org/pdf/2004.09297>`__

    https://github.com/microsoft/MPNet

    **Paper abstract**

    *BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models.
     Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for
     pre-training to address this problem. However, XLNet does not leverage the full position information of a sentence
     and thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet,
     a novel pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet
     leverages the dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes
     auxiliary position information as input to make the model see a full sentence and thus reducing the position
     discrepancy (vs. PLM in XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune
     on a variety of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and
     PLM by a large margin, and achieves better results on these tasks compared with previous state-of-the-art
     pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same model setting.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = MPNetEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("mpnet_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["mpnet_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is an example sentence", "Each sentence is converted"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[[0.022502584, -0.078291744, -0.023030775, -0.0051000593, -0.080340415, 0.039...|
    |[[0.041702367, 0.0010974605, -0.015534201, 0.07092203, -0.0017729357, 0.04661...|
    +--------------------------------------------------------------------------------+
    """

    name = "MPNetEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings", java_model=None):
        super(MPNetEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
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
        MPNetEmbeddings
            The restored model
        """
        from sparknlp.internal import _MPNetLoader
        jModel = _MPNetLoader(folder, spark_session._jsparkSession)._java_obj
        return MPNetEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="all_mpnet_base_v2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "all_mpnet_base_v2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MPNetEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MPNetEmbeddings, name, lang, remote_loc)
