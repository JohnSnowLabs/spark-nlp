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
"""Contains classes for XlmRoBertaEmbeddings."""

from sparknlp.common import *


class XlmRoBertaEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasEngine,
                           HasMaxSentenceLengthLimit):
    """The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual
    Representation Learning at Scale` by Alexis Conneau, Kartikay Khandelwal,
    Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzman, Edouard
    Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.

    It is based on Facebook's RoBERTa model released in 2019. It is a large
    multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = XlmRoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"xlm_roberta_base"``, default language is ``"xx"``
    (meaning multi-lingual), if no values are provided. For available pretrained
    models please see the `Models Hub
    <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb>`__.
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
        True
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Notes
    -----
    - XLM-RoBERTa is a multilingual model trained on 100 different languages.
      Unlike some XLM multilingual models, it does not require **lang**
      parameter to understand which language is used, and should be able to
      determine the correct language from the input ids.
    - This implementation is the same as RoBERTa. Refer to
      :class:`.RoBertaEmbeddings` for usage examples as well as the information
      relative to the inputs and outputs.

    References
    ----------
    `Unsupervised Cross-lingual
    Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__

    **Paper Abstract:**

    *This paper shows that pretraining multilingual language models at scale
    leads to significant performance gains for a wide range of cross-lingual
    transfer tasks. We train a Transformer-based masked language model on one
    hundred languages, using more than two terabytes of filtered CommonCrawl
    data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT
    (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average
    accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1
    score on NER. XLM-R performs particularly well on low-resource languages,
    improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the
    previous XLM model. We also present a detailed empirical evaluation of the
    key factors that are required to achieve these gains, including the
    trade-offs between (1) positive transfer and capacity dilution and (2) the
    performance of high and low resource languages at scale. Finally, we show,
    for the first time, the possibility of multilingual modeling without
    sacrificing per-language performance; XLM-Ris very competitive with strong
    monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code,
    data, and models publicly available.*

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
    >>> embeddings = XlmRoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
    |[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
    |[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
    |[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
    |[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
    +--------------------------------------------------------------------------------+
    """

    name = "XlmRoBertaEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings", java_model=None):
        super(XlmRoBertaEmbeddings, self).__init__(
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
        XlmRoBertaEmbeddings
            The restored model
        """
        from sparknlp.internal import _XlmRoBertaLoader
        jModel = _XlmRoBertaLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return XlmRoBertaEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="xlm_roberta_base", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "xlm_roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        XlmRoBertaEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(XlmRoBertaEmbeddings, name, lang, remote_loc)
