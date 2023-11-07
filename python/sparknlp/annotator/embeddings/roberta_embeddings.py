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
"""Contains classes for RoBertaEmbeddings."""

from sparknlp.common import *


class RoBertaEmbeddings(AnnotatorModel,
                        HasEmbeddingsProperties,
                        HasCaseSensitiveProperties,
                        HasStorageRef,
                        HasBatchedAnnotate,
                        HasEngine,
                        HasMaxSentenceLengthLimit):
    """Creates word embeddings using RoBERTa.

    The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT
    Pretraining Approach` by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du,
    Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin
    Stoyanov. It is based on Google's BERT model released in 2018.

    It builds on BERT and modifies key hyperparameters, removing the
    next-sentence pretraining objective and training with much larger
    mini-batches and learning rates.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = RoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"roberta_base"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb>`__.
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
    - RoBERTa has the same architecture as BERT, but uses a byte-level BPE as
      a tokenizer (same as GPT-2) and uses a different pretraining scheme.
    - RoBERTa doesn't have ``token_type_ids``, you don't need to indicate
      which token belongs to which segment. Just separate your segments with
      the separation token ``tokenizer.sep_token`` (or ``</s>``)

    References
    ----------
    `RoBERTa: A Robustly Optimized BERT
    Pretraining Approach <https://arxiv.org/abs/1907.11692>`__

    **Paper Abstract:**

    *Language model pretraining has led to significant performance gains but
    careful comparison between different approaches is challenging. Training is
    computationally expensive, often done on private datasets of different
    sizes, and, as we will show, hyperparameter choices have significant impact
    on the final results. We present a replication study of BERT pretraining
    (Devlin et al., 2019) that carefully measures the impact of many key
    hyperparameters and training data size. We find that BERT was significantly
    undertrained, and can match or exceed the performance of every model
    published after it. Our best model achieves state-of-the-art results on
    GLUE, RACE and SQuAD. These results highlight the importance of previously
    overlooked design choices, and raise questions about the source of recently
    reported improvements. We release our models and code.*

    Source of the original code: `RoBERTa: A Robustly Optimized BERT Pretraining
    Approach on GitHub
    <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`__.

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
    >>> embeddings = RoBertaEmbeddings.pretrained() \\
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
    |[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
    |[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
    |[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
    |[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
    |[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
    +--------------------------------------------------------------------------------+
    """

    name = "RoBertaEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings", java_model=None):
        super(RoBertaEmbeddings, self).__init__(
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
        RoBertaEmbeddings
            The restored model
        """
        from sparknlp.internal import _RoBertaLoader
        jModel = _RoBertaLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return RoBertaEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="roberta_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        RoBertaEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(RoBertaEmbeddings, name, lang, remote_loc)
