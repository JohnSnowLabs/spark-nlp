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
"""Contains classes for the T5Transformer."""

from sparknlp.common import *


class T5Transformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """T5: the Text-To-Text Transfer Transformer

    T5 reconsiders all NLP tasks into a unified text-to-text-format where the
    input and output are always text strings, in contrast to BERT-style models
    that can only output either a class label or a span of the input. The
    text-to-text framework is able to use the same model, loss function, and
    hyper-parameters on any NLP task, including machine translation, document
    summarization, question answering, and classification tasks (e.g., sentiment
    analysis). T5 can even apply to regression tasks by training it to predict
    the string representation of a number instead of the number itself.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> t5 = T5Transformer.pretrained() \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("summaries")


    The default model is ``"t5_small"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=t5>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/question-answering/Question_Answering_and_Summarization_with_T5.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    task
        Transformer's task, e.g. ``summarize:``
    minOutputLength
        Minimum length of the sequence to be generated
    maxOutputLength
        Maximum length of output text
    doSample
        Whether or not to use sampling; use greedy decoding otherwise
    temperature
        The value used to module the next token probabilities
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering
    topP
        Top cumulative probability for vocabulary tokens

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty.
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once
    ignoreTokenIds
       A list of token ids which are ignored in the decoder's output

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    - `Exploring Transfer Learning with T5: the Text-To-Text Transfer
      Transformer
      <https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html>`__
    - `Exploring the Limits of Transfer Learning with a Unified Text-to-Text
      Transformer <https://arxiv.org/abs/1910.10683>`__
    - https://github.com/google-research/text-to-text-transfer-transformer

    **Paper Abstract:**

    *Transfer learning, where a model is first pre-trained on a data-rich task
    before being fine-tuned on a downstream task, has emerged as a powerful
    technique in natural language processing (NLP). The effectiveness of
    transfer learning has given rise to a diversity of approaches, methodology,
    and practice. In this paper, we explore the landscape of transfer learning
    techniques for NLP by introducing a unified framework that converts all
    text-based language problems into a text-to-text format. Our systematic
    study compares pre-training objectives, architectures, unlabeled data sets,
    transfer approaches, and other factors on dozens of language understanding
    tasks. By combining the insights from our exploration with scale and our new
    Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many
    benchmarks covering summarization, question answering, text classification,
    and more. To facilitate future work on transfer learning for NLP, we release
    our data set, pre-trained models, and code.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> t5 = T5Transformer.pretrained("t5_small") \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(200) \\
    ...     .setOutputCol("summaries")
    >>> pipeline = Pipeline().setStages([documentAssembler, t5])
    >>> data = spark.createDataFrame([[
    ...     "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
    ...     "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
    ...     " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
    ...     "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
    ...     "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
    ...     "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
    ...     "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
    ...     "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
    ...     "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
    ...     "learning for NLP, we release our data set, pre-trained models, and code."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.result").show(truncate=False)
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                                        |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "T5Transformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    task = Param(Params._dummy(), "task", "Transformer's task, e.g. summarize>", typeConverter=TypeConverters.toString)

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

    useCache = Param(Params._dummy(), "useCache", "Cache internal state of the model to improve performance",
                     typeConverter=TypeConverters.toBoolean)

    stopAtEos = Param(
        Params._dummy(),
        "stopAtEos",
        "Stop text generation when the end-of-sentence token is encountered.",
        typeConverter=TypeConverters.toBoolean
    )

    maxNewTokens = Param(
        Params._dummy(),
        "maxNewTokens",
        "Maximum number of new tokens to be generated",
        typeConverter=TypeConverters.toInt
    )

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

    def setTask(self, value):
        """Sets the transformer's task, e.g. ``summarize:``.

        Parameters
        ----------
        value : str
            The transformer's task
        """
        return self._set(task=value)

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

    def setStopAtEos(self, b):
        """Stop text generation when the end-of-sentence token is encountered.

        Parameters
        ----------
        b : bool
            whether to stop at end-of-sentence token or not
        """
        return self._set(stopAtEos=b)

    def setMaxNewTokens(self, value):
        """Sets the maximum number of new tokens to be generated

        Parameters
        ----------
        value : int
            the maximum number of new tokens to be generated
        """
        return self._set(maxNewTokens=value)

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

    def setUseCache(self, value):
        """Cache internal state of the model to improve performance

        Parameters
        ----------
        value : bool
            Whether or not to use cache
        """
        return self._set(useCache=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer", java_model=None):
        super(T5Transformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            task="",
            minOutputLength=0,
            maxOutputLength=20,
            doSample=False,
            temperature=1.0,
            topK=50,
            topP=1.0,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[],
            batchSize=1,
            stopAtEos=True,
            maxNewTokens=512,
            useCache=False
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
        T5Transformer
            The restored model
        """
        from sparknlp.internal import _T5Loader
        jModel = _T5Loader(folder, spark_session._jsparkSession)._java_obj
        return T5Transformer(java_model=jModel)

    @staticmethod
    def pretrained(name="t5_small", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "t5_small"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        T5Transformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(T5Transformer, name, lang, remote_loc)
