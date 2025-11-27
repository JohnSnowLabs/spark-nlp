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
"""Contains classes for DeBertaEmbeddings."""
from sparknlp.common import *


class DeBertaEmbeddings(AnnotatorModel,
                        HasEmbeddingsProperties,
                        HasCaseSensitiveProperties,
                        HasStorageRef,
                        HasBatchedAnnotate,
                        HasEngine,
                        HasMaxSentenceLengthLimit):
    """The DeBERTa model was proposed in DeBERTa: Decoding-enhanced BERT with
    Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu
    Chen It is based on Google’s BERT model released in 2018 and Facebook’s
    RoBERTa model released in 2019.

    This model requires input tokenization with
    SentencePiece model, which is provided by Spark NLP (See tokenizers
    package).

    It builds on RoBERTa with disentangled attention and enhanced mask decoder
    training with half of the data used in RoBERTa.

    Pretrained models can be loaded with pretrained of the companion object:

    >>> embeddings = DeBertaEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence", "token"]) \\
    ...    .setOutputCol("embeddings")

    The default model is ``"deberta_v3_base"``, if no name is provided.

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
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128

    References
    ----------
    https://github.com/microsoft/DeBERTa

    https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/

    **Paper abstract:**

    *Paper abstract: Recent progress in pre-trained neural language models has
    significantly improved the performance of many natural language processing
    (NLP) tasks. In this paper we propose a new model architecture DeBERTa (
    Decoding-enhanced BERT with disentangled attention) that improves the BERT
    and RoBERTa models using two novel techniques. The first is the disentangled
    attention mechanism, where each word is represented using two vectors that
    encode its content and position, respectively, and the attention weights
    among words are computed using disentangled matrices on their contents and
    relative positions. Second, an enhanced mask decoder is used to replace the
    output softmax layer to predict the masked tokens for model pretraining. We
    show that these two techniques significantly improve the efficiency of model
    pretraining and performance of downstream tasks. Compared to RoBERTa-Large,
    a DeBERTa model trained on half of the training data performs consistently
    better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9%
    (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by
    +3.6% (83.2% vs. 86.8%). The DeBERTa code and pre-trained models will be
    made publicly available at https://github.com/microsoft/DeBERTa.*

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
    >>> embeddings = DeBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
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
    |[1.1342473030090332,-1.3855540752410889,0.9818322062492371,-0.784737348556518...|
    |[0.847029983997345,-1.047153353691101,-0.1520637571811676,-0.6245765686035156...|
    |[-0.009860038757324219,-0.13450059294700623,2.707749128341675,1.2916892766952...|
    |[-0.04192575812339783,-0.5764210224151611,-0.3196685314178467,-0.527840495109...|
    |[0.15583214163780212,-0.1614152491092682,-0.28423872590065,-0.135491415858268...|
    +--------------------------------------------------------------------------------+
    """

    name = "DeBertaEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.DeBertaEmbeddings", java_model=None):
        super(DeBertaEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            dimension=768,
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
        DeBertaEmbeddings
            The restored model
        """
        from sparknlp.internal import _DeBERTaLoader
        jModel = _DeBERTaLoader(folder, spark_session._jsparkSession)._java_obj
        return DeBertaEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="deberta_v3_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "deberta_v3_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DeBertaEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DeBertaEmbeddings, name, lang, remote_loc)
