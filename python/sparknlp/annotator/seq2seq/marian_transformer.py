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
"""Contains classes for the MarianTransformer."""

from sparknlp.common import *


class MarianTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """MarianTransformer: Fast Neural Machine Translation

    Marian is an efficient, free Neural Machine Translation framework written in
    pure C++ with minimal dependencies. It is mainly being developed by the
    Microsoft Translator team. Many academic (most notably the University of
    Edinburgh and in the past the Adam Mickiewicz University in Poznań) and
    commercial contributors help with its development. MarianTransformer uses
    the models trained by MarianNMT.

    It is currently the engine behind the Microsoft Translator Neural Machine
    Translation services and being deployed by many companies, organizations and
    research projects.

    Note that this model only supports inputs up to 512 tokens. If you are
    working with longer inputs, consider splitting them first. For example, you
    can use the SentenceDetectorDL annotator to split longer texts into
    sentences.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> marian = MarianTransformer.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("translation")

    The default model is ``"opus_mt_en_fr"``, default language is ``"xx"``
    (meaning multi-lingual), if no values are provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Translation>`__.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/multilingual/Translation_Marian.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 1
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    langId
        Transformer's task, e.g. "summarize>", by default ""
    maxInputLength
        Controls the maximum length for encoder inputs (source language texts),
        by default 40
    maxOutputLength
        Controls the maximum length for decoder outputs (target language texts),
        by default 40

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    `MarianNMT at GitHub <https://marian-nmt.github.io/>`__

    `Marian: Fast Neural Machine Translation in C++  <https://www.aclweb.org/anthology/P18-4020/>`__

    **Paper Abstract:**

    *We present Marian, an efficient and self-contained Neural Machine
    Translation framework with an integrated automatic differentiation
    engine based on dynamic computation graphs. Marian is written entirely in
    C++. We describe the design of the encoder-decoder framework and
    demonstrate that a research-friendly toolkit can achieve high training
    and translation speed.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence")
    >>> marian = MarianTransformer.pretrained() \\
    ...     .setInputCols("sentence") \\
    ...     .setOutputCol("translation") \\
    ...     .setMaxInputLength(30)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       marian
    ...     ])
    >>> data = spark.createDataFrame([["What is the capital of France? We should know this in french."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(translation.result) as result").show(truncate=False)
    +-------------------------------------+
    |result                               |
    +-------------------------------------+
    |Quelle est la capitale de la France ?|
    |On devrait le savoir en français.    |
    +-------------------------------------+
    """

    name = "MarianTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    langId = Param(Params._dummy(), "langId", "Transformer's task, e.g. summarize>",
                   typeConverter=TypeConverters.toString)

    maxInputLength = Param(Params._dummy(), "maxInputLength",
                           "Controls the maximum length for encoder inputs (source language texts)",
                           typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength",
                            "Controls the maximum length for decoder outputs (target language texts)",
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

    def setLangId(self, value):
        """Sets transformer's task, e.g. "summarize>", by default "".

        Parameters
        ----------
        value : str
            Transformer's task, e.g. "summarize>"
        """
        return self._set(langId=value)

    def setMaxInputLength(self, value):
        """Sets the maximum length for encoder inputs (source language texts),
        by default 40. The value should be less than 512, as the Marian Transformer does not
        support inputs longer than 512 tokens.

        Parameters
        ----------
        value : int
            The maximum length for encoder inputs (source language texts)
        """
        if value > 512:
            raise ValueError("MarianTransformer model does not support sequences longer than 512.")
        return self._set(maxInputLength=value)

    def setMaxOutputLength(self, value):
        """Sets the maximum length for decoder outputs (target language texts),
        by default 40.

        Parameters
        ----------
        value : int
            The maximum length for decoder outputs (target language texts)
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

    def setRandomSeed(self, seed):
        """Sets random seed.

        Parameters
        ----------
        seed : int
            Random seed
        """
        self._call_java("setRandomSeed", seed)

        return self

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer", java_model=None):
        super(MarianTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=1,
            maxInputLength=40,
            maxOutputLength=40,
            langId="",
            doSample=False,
            temperature=1.0,
            topK=50,
            topP=1.0,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[]
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
        MarianTransformer
            The restored model
        """
        from sparknlp.internal import _MarianLoader
        jModel = _MarianLoader(folder, spark_session._jsparkSession)._java_obj
        return MarianTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="opus_mt_en_fr", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "opus_mt_en_fr"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MarianTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MarianTransformer, name, lang, remote_loc)
