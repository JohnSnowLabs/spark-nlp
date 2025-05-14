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
"""Contains classes for the CoHereTransformer."""

from sparknlp.common import *


class CoHereTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """Cohere: Command-R Transformer
    
        C4AI Command-R is a research release of a 35 billion parameter highly performant generative model.
        Command-R is a large language model with open weights optimized for a variety of use cases including reasoning,
        summarization, and question answering. Command-R has the capability for multilingual generation evaluated
        in 10 languages and highly performant RAG capabilities.

        Pretrained models can be loaded with :meth:`.pretrained` of the companion
        object:
    
        >>> CoHere = CoHereTransformer.pretrained() \\
        ...     .setInputCols(["document"]) \\
        ...     .setOutputCol("generation")
    
    
        The default model is ``"c4ai_command_r_v01_int4"``, if no name is provided. For available
        pretrained models please see the `Models Hub
        <https://sparknlp.org/models?q=CoHere>`__.
    
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
            Maximum length of output text, by default 60
        doSample
            Whether or not to use sampling; use greedy decoding otherwise, by default False
        temperature
            The value used to modulate the next token probabilities, by default 1.0
        topK
            The number of highest probability vocabulary tokens to keep for
            top-k-filtering, by default 40
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
        This is a very computationally expensive module, especially on larger
        sequences. The use of an accelerator such as GPU is recommended.
    
        References
        ----------
        - `Cohere <https://cohere.for.ai/>`__

    
        Examples
        --------
        >>> import sparknlp
        >>> from sparknlp.base import *
        >>> from sparknlp.annotator import *
        >>> from pyspark.ml import Pipeline
        >>> documentAssembler = DocumentAssembler() \\
        ...     .setInputCol("text") \\
        ...     .setOutputCol("documents")
        >>> CoHere = CoHereTransformer.pretrained("c4ai_command_r_v01_int4","en") \\
        ...     .setInputCols(["documents"]) \\
        ...     .setMaxOutputLength(60) \\
        ...     .setOutputCol("generation")
        >>> pipeline = Pipeline().setStages([documentAssembler, CoHere])
        >>> data = spark.createDataFrame([
        ...     (
        ...         1,
        ...         "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ...         )
        ... ]).toDF("id", "text")
        >>> result = pipeline.fit(data).transform(data)
        >>> result.select("generation.result").show(truncate=False)
        +------------------------------------------------+
        |result                                          |
        +------------------------------------------------+
        |[Hello! I'm doing well, thank you for asking! I'm excited to help you with whatever questions you have today. How can I assist you?]|
        +------------------------------------------------+
    """

    name = "CoHereTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
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

    beamSize = Param(Params._dummy(), "beamSize",
                     "The number of beams to use for beam search",
                     typeConverter=TypeConverters.toInt)

    stopTokenIds = Param(Params._dummy(), "stopTokenIds",
                         "A list of token ids which are considered as stop tokens in the decoder's output",
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

    def setBeamSize(self, value):
        """Sets the number of beams to use for beam search.

        Parameters
        ----------
        value : int
            The number of beams to use for beam search
        """
        return self._set(beamSize=value)

    def setStopTokenIds(self, value):
        """Sets a list of token ids which are considered as stop tokens in the decoder's output.

        Parameters
        ----------
        value : List[int]
            The words to be considered as stop tokens
        """
        return self._set(stopTokenIds=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.CoHereTransformer", java_model=None):
        super(CoHereTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            minOutputLength=0,
            maxOutputLength=20,
            doSample=False,
            temperature=0.6,
            topK=-1,
            topP=0.9,
            repetitionPenalty=1.0,
            noRepeatNgramSize=3,
            ignoreTokenIds=[],
            batchSize=1,
            beamSize=1,
            stopTokenIds=[128001, ]
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

        Returns
        -------
        CoHereTransformer
            The restored model
        """
        from sparknlp.internal import _CoHereLoader
        jModel = _CoHereLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return CoHereTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="c4ai_command_r_v01_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "c4ai_command_r_v01_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CoHereTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CoHereTransformer, name, lang, remote_loc)
