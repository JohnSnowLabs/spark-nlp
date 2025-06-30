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
"""Contains classes for the QwenTransformer."""

from sparknlp.common import *


class QwenTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """Qwen: comprehensive language model series

   Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model
   pretrained on a large amount of data. In comparison with the previous released Qwen, the
   improvements include:

   6 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, and 72B; Significant performance improvement
   in Chat models; Multilingual support of both base and chat models; Stable support of 32K
   context length for models of all sizes

   Qwen1.5 is a language model series including decoder language models of different model sizes.
   For each size, we release the base language model and the aligned chat model. It is based on
   the Transformer architecture with SwiGLU activation, attention QKV bias, group query
   attention, mixture of sliding window attention and full attention, etc. Additionally, we have
   an improved tokenizer adaptive to multiple natural languages and codes. For the beta version,
   temporarily we did not include GQA and the mixture of SWA and full attention.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> qwen = QwenTransformer.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("generation")


    The default model is ``"qwen-13b"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=qwen>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    **References**
    
    - `Qwen Technical Report
      <https://arxiv.org/pdf/2309.16609.pdf>`__
    - https://qwenlm.github.io/blog/qwen1.5/
    - https://github.com/QwenLM/Qwen1.5

    **Paper Abstract:**

    *Large language models (LLMs) have revolutionized the field of artificial intelligence,
    enabling natural language processing tasks that were previously thought to be exclusive to
    humans. In this work, we introduce Qwen, the first installment of our large language model
    series. Qwen is a comprehensive language model series that encompasses distinct models with
    varying parameter counts. It includes Qwen, the base pretrained language models, and
    Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models
    consistently demonstrate superior performance across a multitude of downstream tasks, and the
    chat models, particularly those trained using Reinforcement Learning from Human Feedback
    (RLHF), are highly competitive. The chat models possess advanced tool-use and planning
    capabilities for creating agent applications, showcasing impressive performance even when
    compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we
    have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as
    mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These
    models demonstrate significantly improved performance in comparison with open-source models,
    and slightly fall behind the proprietary models.*


    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    minOutputLength
        Minimum length of the sequence to be generated, by default 0
    maxOutputLength
        Maximum length of output text, by default 20
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default False
    temperature
        The value used to module the next token probabilities, by default 1.0
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering, by default 50
    topP
        Top cumulative probability for vocabulary tokens, by default 1.0

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty, 1.0 means no penalty. , by default
        1.0
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by
        default 0
    ignoreTokenIds
        A list of token ids which are ignored in the decoder's output, by
        default []

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> qwen = QwenTransformer.pretrained("qwen_7.5b_chat") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(50) \\
    ...     .setOutputCol("generation")
    >>> pipeline = Pipeline().setStages([documentAssembler, qwen])
    >>> data = spark.createDataFrame([["My name is Leonardo."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.generation").show(truncate=False)
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                              |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[My name is Leonardo . I am a student of the University of California, Berkeley. I am interested in the field of Artificial Intelligence and its applications in the real world. I have a strong    |
    | passion for learning and am always looking for ways to improve my knowledge and skills]                                                                                                            |
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "QwenTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated",
                            typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text",
                            typeConverter=TypeConverters.toInt)

    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise",
                     typeConverter=TypeConverters.toBoolean)

    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities",
                        typeConverter=TypeConverters.toFloat)

    topK = Param(Params._dummy(), "topK",
                 "The number of highest probability vocabulary tokens to keep for top-k-filtering",
                 typeConverter=TypeConverters.toInt)

    topP = Param(Params._dummy(), "topP",
                 "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation",
                 typeConverter=TypeConverters.toFloat)

    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details",
                              typeConverter=TypeConverters.toFloat)

    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize",
                              "If set to int > 0, all ngrams of that size can only occur once",
                              typeConverter=TypeConverters.toInt)

    ignoreTokenIds = Param(Params._dummy(), "ignoreTokenIds",
                           "A list of token ids which are ignored in the decoder's output",
                           typeConverter=TypeConverters.toListInt)

    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output.

        Parameters
        ----------
        value : List[int]
            The words to be filtered out
        """
        return self._set(ignoreTokenIds=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty.

        Parameters
        ----------
        value : float
            The repetition penalty

        References
        ----------
        See `Ctrl: A Conditional Transformer Language Model For Controllable
        Generation <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        """
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.QwenTransformer", java_model=None):
        super(QwenTransformer, self).__init__(classname=classname, java_model=java_model)
        self._setDefault(minOutputLength=0, maxOutputLength=50, doSample=False, temperature=0.6, topK=50, topP=0.9,
            repetitionPenalty=1.0, noRepeatNgramSize=0, ignoreTokenIds=[], batchSize=1)

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        QwenTransformer
            The restored model
        """
        from sparknlp.internal import _QwenLoader
        jModel = _QwenLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return QwenTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="qwen_7.5b_chat", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "qwen_7.5b_chat"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        QwenTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(QwenTransformer, name, lang, remote_loc)
