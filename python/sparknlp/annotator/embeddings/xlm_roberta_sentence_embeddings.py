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
"""Contains classes for XlmRoBertaSentenceEmbeddings."""

from sparknlp.common import *


class XlmRoBertaSentenceEmbeddings(AnnotatorModel,
                                        HasEmbeddingsProperties,
                                        HasCaseSensitiveProperties,
                                        HasStorageRef,
                                        HasBatchedAnnotate,
                                        HasEngine,
                                        HasMaxSentenceLengthLimit):
    """Sentence-level embeddings using XLM-RoBERTa. The XLM-RoBERTa model was proposed in Unsupervised Cross-lingual
    Representation Learning at Scale  by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary,
    Guillaume Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based
    on Facebook's RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of
    filtered CommonCrawl data. Pretrained models can be loaded with pretrained of the companion object:

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = XlmRoBertaSentenceEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence"]) \\
    ...    .setOutputCol("sentence_embeddings")


    The default model is ``"sent_xlm_roberta_base"``, if no name is provided.

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
    `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/pdf/1911.02116.pdf>`__

    **Paper abstract:**

    *This paper shows that pretraining multilingual language models at scale leads to significant performance gains
    for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one
    hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R,
    significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8%
    average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs
    particularly well on low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over
    the previous XLM model. We also present a detailed empirical evaluation of the key factors that are required to
    achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the
    performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of
    multilingual modeling without sacrificing per-language performance; XLM-Ris very competitive with strong
    monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code, data, and models publicly available.*

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
    >>> embeddings = XlmRoBertaSentenceEmbeddings.pretrained() \\
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

    name = "XlmRoBertaSentenceEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings", java_model=None):
        super(XlmRoBertaSentenceEmbeddings, self).__init__(
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
        from sparknlp.internal import _XlmRoBertaSentenceLoader
        jModel = _XlmRoBertaSentenceLoader(folder, spark_session._jsparkSession)._java_obj
        return XlmRoBertaSentenceEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="sent_xlm_roberta_base", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sent_xlm_roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        XlmRoBertaSentenceEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(XlmRoBertaSentenceEmbeddings, name, lang, remote_loc)
