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
"""Contains classes for the StarCoderTransformer."""

from sparknlp.common import *


class StarCoderTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """StarCoder2: The Versatile Code Companion.

    StarCoder2 is a Transformer model designed specifically for code generation and understanding.
    With 13 billion parameters, it builds upon the advancements of its predecessors and is trained
    on a diverse dataset that includes multiple programming languages. This extensive training
    allows StarCoder2 to support a wide array of coding tasks, from code completion to generation.

    StarCoder2 was developed to assist developers in writing and understanding code more efficiently,
    making it a valuable tool for various software development and data science tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> starcoder2 = StarCoder2Transformer.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("generation")

    The default model is ``"starcoder2-13b"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=starcoder2>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

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
        The value used to modulate the next token probabilities, by default 1.0
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

    References
    ----------
    - `StarCoder2: The Versatile Code Companion.
      <https://huggingface.co/blog/starcoder>`__
    - https://github.com/bigcode-project/starcoder

    **Paper Abstract:**

    *The BigCode project, an open-scientific collaboration focused on the responsible
    development of Large Language Models for Code (Code LLMs), introduces StarCoder2. In
    partnership with Software Heritage (SWH), we build The Stack v2 on top of the digital commons
    of their source code archive. Alongside the SWH repositories spanning 619 programming
    languages, we carefully select other high-quality data sources, such as GitHub pull requests,
    Kaggle notebooks, and code documentation. This results in a training set that is 4× larger
    than the first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters
    on 3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM
    benchmarks.*

    *We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on
    most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2-15B,
    significantly outperforms other models of comparable size. In addition, it matches or
    outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder-33B is
    the best-performing model at code completion for high-resource languages, we find that
    StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several
    low-resource languages. We make the model weights available under an OpenRAIL license and
    ensure full transparency regarding the training data by releasing the Software Heritage
    persistent Identifiers (SWHIDs) of the source code data.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> starcoder2 = StarCoder2Transformer.pretrained("starcoder2") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(50) \\
    ...     .setOutputCol("generation")
    >>> pipeline = Pipeline().setStages([documentAssembler, starcoder2])
    >>> data = spark.createDataFrame([["def add(a, b):"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("generation.result").show(truncate=False)
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                              |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[def add(a, b): return a + b]                                                                                                                                                                       |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """



    name = "StarCoderTransformer"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.StarCoderTransformer", java_model=None):
        super(StarCoderTransformer, self).__init__(classname=classname, java_model=java_model)
        self._setDefault(minOutputLength=0, maxOutputLength=20, doSample=False, temperature=0.6, topK=50, topP=0.9,
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
        StarCoderTransformer
            The restored model
        """
        from sparknlp.internal import _StarCoderLoader
        jModel = _StarCoderLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return StarCoderTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="starcoder", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "starcoder"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        StarCoderTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(StarCoderTransformer, name, lang, remote_loc)
