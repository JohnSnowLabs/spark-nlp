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
"""Contains classes for the Phi3Transformer."""

from sparknlp.common import *


class Phi3Transformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """Phi-3
    
    The Phi-3-Medium-4K-Instruct is a 14B parameters, lightweight, state-of-the-art open model trained with the Phi-3
    datasets that includes both synthetic data and the filtered publicly available websites data with a focus on
    high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Medium version in two
    variants 4K and 128K which is the context length (in tokens) that it can support.

    The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference
    optimization for the instruction following and safety measures. When assessed against benchmarks testing common
    sense, language understanding, math, code, long context and logical reasoning, Phi-3-Medium-4K-Instruct showcased
    a robust and state-of-the-art performance among models of the same-size and next-size-up.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> phi3 = Phi3Transformer.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("generation")


    The default model is ``phi_3_mini_128k_instruct``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=phi3>`__.

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

    References
    ----------
    - `Phi-3: Small Language Models with Big Potential
      <https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential//>`__
    - https://huggingface.co/microsoft/phi-3

    **Paper Abstract:**

    *We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion
    tokens, whose overall performance, as measured by both academic benchmarks and internal
    testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69%
    on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. The
    innovation lies entirely in our dataset for training, a scaled-up version of the one used for
    phi-2, composed of heavily filtered publicly available web data and synthetic data. The model
    is also further aligned for robustness, safety, and chat format. We also provide some initial
    parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small
    and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and
    78% on MMLU, and 8.7 and 8.9 on MT-bench). Moreover, we also introduce phi-3-vision, a 4.2
    billion parameter model based on phi-3-mini with strong reasoning capabilities for image and
    text prompts.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> phi3 = Phi3Transformer.pretrained(phi_3_mini_128k_instruct) \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(50) \\
    ...     .setOutputCol("generation")
    >>> pipeline = Pipeline().setStages([documentAssembler, phi3])
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

    name = "Phi3Transformer"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.Phi3Transformer", java_model=None):
        super(Phi3Transformer, self).__init__(classname=classname, java_model=java_model)
        self._setDefault(minOutputLength=0, maxOutputLength=20, doSample=False, temperature=1.0, topK=500, topP=1.0,
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
        Phi3Transformer
            The restored model
        """
        from sparknlp.internal import _Phi3Loader
        jModel = _Phi3Loader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return Phi3Transformer(java_model=jModel)

    @staticmethod
    def pretrained(name="phi_3_mini_128k_instruct", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default phi_3_mini_128k_instruct
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        Phi3Transformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(Phi3Transformer, name, lang, remote_loc)
