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
"""Contains classes for ElmoEmbeddings."""

from sparknlp.common import *


class ElmoEmbeddings(AnnotatorModel,
                     HasEmbeddingsProperties,
                     HasCaseSensitiveProperties,
                     HasStorageRef,
                     HasEngine):
    """Word embeddings from ELMo (Embeddings from Language Models), a language
    model trained on the 1 Billion Word Benchmark.

    Note that this is a very computationally expensive module compared to word
    embedding modules that only perform embedding lookups. The use of an
    accelerator is recommended.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = ElmoEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("elmo_embeddings")


    The default model is ``"elmo"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

    The pooling layer can be set with :meth:`.setPoolingLayer` to the following
    values:

    - ``"word_emb"``: the character-based word representations with shape
      ``[batch_size, max_length, 512]``.
    - ``"lstm_outputs1"``: the first LSTM hidden state with shape
      ``[batch_size, max_length, 1024]``.
    - ``"lstm_outputs2"``: the second LSTM hidden state with shape
      ``[batch_size, max_length, 1024]``.
    - ``"elmo"``: the weighted sum of the 3 layers, where the weights are
      trainable. This tensor has shape ``[batch_size, max_length, 1024]``.

    For extended examples of usage, see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_elmo.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 32
    dimension
        Number of embedding dimensions
    caseSensitive
        Whether to ignore case in tokens for embeddings matching
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    poolingLayer
        Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or
        elmo, by default word_emb

    References
    ----------
    https://tfhub.dev/google/elmo/3

    `Deep contextualized word representations <https://arxiv.org/abs/1802.05365>`__

    **Paper abstract:**

    *We introduce a new type of deep contextualized word representation that
    models both (1) complex characteristics of word use (e.g., syntax and
    semantics), and (2) how these uses vary across linguistic contexts (i.e.,
    to model polysemy). Our word vectors are learned functions of the internal
    states of a deep bidirectional language model (biLM), which is pre-trained
    on a large text corpus. We show that these representations can be easily
    added to existing models and significantly improve the state of the art
    across six challenging NLP problems, including question answering, textual
    entailment and sentiment analysis. We also present an analysis showing that
    exposing the deep internals of the pre-trained network is crucial, allowing
    downstream models to mix different types of semi-supervision signals.*

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
    >>> embeddings = ElmoEmbeddings.pretrained() \\
    ...     .setPoolingLayer("word_emb") \\
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
    |[6.662458181381226E-4,-0.2541114091873169,-0.6275503039360046,0.5787073969841...|
    |[0.19154725968837738,0.22998669743537903,-0.2894386649131775,0.21524395048618...|
    |[0.10400570929050446,0.12288510054349899,-0.07056470215320587,-0.246389418840...|
    |[0.49932169914245605,-0.12706467509269714,0.30969417095184326,0.2643227577209...|
    |[-0.8871506452560425,-0.20039963722229004,-1.0601330995559692,0.0348707810044...|
    +--------------------------------------------------------------------------------+
    """

    name = "ElmoEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    poolingLayer = Param(Params._dummy(),
                         "poolingLayer", "Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo",
                         typeConverter=TypeConverters.toString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setBatchSize(self, value):
        """Sets batch size, by default 32.

        Parameters
        ----------
        value : int
            Batch size
        """
        return self._set(batchSize=value)

    def setPoolingLayer(self, layer):
        """Sets ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or
        elmo, by default word_emb

        Parameters
        ----------
        layer : str
            ELMO pooling layer
        """
        if layer == "word_emb":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs1":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs2":
            return self._set(poolingLayer=layer)
        elif layer == "elmo":
            return self._set(poolingLayer=layer)
        else:
            return self._set(poolingLayer="word_emb")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings", java_model=None):
        super(ElmoEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=32,
            poolingLayer="word_emb"
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
        ElmoEmbeddings
            The restored model
        """
        from sparknlp.internal import _ElmoLoader
        jModel = _ElmoLoader(folder, spark_session._jsparkSession)._java_obj
        return ElmoEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="elmo", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "elmo"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ElmoEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ElmoEmbeddings, name, lang, remote_loc)
