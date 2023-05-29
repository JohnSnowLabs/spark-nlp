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
"""Contains classes for LongformerEmbeddings."""

from sparknlp.common import *


class LongformerEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasEngine,
                           HasLongMaxSentenceLengthLimit):
    """Longformer is a transformer model for long documents. The Longformer
    model was presented in `Longformer: The Long-Document Transformer` by Iz
    Beltagy, Matthew E. Peters, Arman Cohan. longformer-base-4096 is a BERT-like
    model started from the RoBERTa checkpoint and pretrained for MLM on long
    documents. It supports sequences of length up to 4,096.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = LongformerEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")


    The default model is ``"longformer_base_4096"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Embeddings>`__.

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
        Max sentence length to process, by default 1024
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Longformer: The Long-Document Transformer
    <https://arxiv.org/pdf/2004.05150.pdf>`__


    **Paper Abstract:**

    *Transformer-based models are unable to process long sequences due to their
    self-attention operation, which scales quadratically with the sequence
    length. To address this limitation, we introduce the Longformer with an
    attention mechanism that scales linearly with sequence length, making it
    easy to process documents of thousands of tokens or longer. Longformer's
    attention mechanism is a drop-in replacement for the standard self-attention
    and combines a local windowed attention with a task motivated global
    attention. Following prior work on long-sequence transformers, we evaluate
    Longformer on character-level language modeling and achieve state-of-the-art
    results on text8 and enwik8. In contrast to most prior work, we also
    pretrain Longformer and finetune it on a variety of downstream tasks. Our
    pretrained Longformer consistently outperforms RoBERTa on long document
    tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We
    finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant
    for supporting long document generative sequence-to-sequence tasks, and
    demonstrate its effectiveness on the arXiv summarization dataset.*

    The original code can be found at `Longformer: The Long-Document Transformer
    <https://github.com/allenai/longformer>`__.

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
    >>> embeddings = LongformerEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    >>>     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...         documentAssembler,
    ...         tokenizer,
    ...         embeddings,
    ...         embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
    |[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
    |[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
    |[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
    |[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
    +--------------------------------------------------------------------------------+
    """
    name = "LongformerEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings", java_model=None):
        super(LongformerEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=1024,
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
        LongformerEmbeddings
            The restored model
        """
        from sparknlp.internal import _LongformerLoader
        jModel = _LongformerLoader(folder, spark_session._jsparkSession)._java_obj
        return LongformerEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="longformer_base_4096", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "longformer_base_4096"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LongformerEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LongformerEmbeddings, name, lang, remote_loc)
