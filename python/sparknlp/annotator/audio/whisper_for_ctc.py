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

"""Contains classes concerning WhisperForCTC."""

from sparknlp.common import *


class WhisperForCTC(AnnotatorModel,
                    HasBatchedAnnotateAudio,
                    HasAudioFeatureProperties,
                    HasEngine, HasGeneratorProperties):
    """Whisper Model with a language modeling head on top for Connectionist Temporal Classification
    (CTC).

    Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of
    multilingual and multitask supervised data collected from the web. It transcribe in multiple
    languages, as well as translate from those languages into English.

    The audio needs to be provided pre-processed an array of floats.

    Note that at the moment, this annotator only supports greedy search and only Spark Versions
    3.4 and up are supported.

    For multilingual models, the language and the task (transcribe or translate) can be set with
    ``setLanguage`` and ``setTask``.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

        speechToText = WhisperForCTC.pretrained() \\
            .setInputCols(["audio_assembler"]) \\
            .setOutputCol("text")


    The default model is ``"asr_whisper_tiny_opt"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    To see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `WhisperForCTCTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTCTest.scala>`__.

    **References:**

    `Robust Speech Recognition via Large-Scale Weak Supervision <https://arxiv.org/abs/2212.04356>`__

    **Paper Abstract:**

    *We study the capabilities of speech processing systems trained simply to predict large
    amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual
    and multitask supervision, the resulting models generalize well to standard benchmarks and are
    often competitive with prior fully supervised results but in a zero- shot transfer setting
    without the need for any fine- tuning. When compared to humans, the models approach their
    accuracy and robustness. We are releasing models and inference code to serve as a foundation
    for further work on robust speech processing.*

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``AUDIO``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    task
        The formatted task for the audio. Either `<|translate|>` or `<|transcribe|>`.
    language
        The language for the audio, formatted to e.g. `<|en|>`. Check the model description for
        supported languages.
    isMultilingual
        Whether the model is multilingual
    minOutputLength
        Minimum length of the sequence to be generated
    maxOutputLength
        Maximum length of output text
    doSample
        Whether or not to use sampling; use greedy decoding otherwise
    temperature
        The value used to module the next token probabilities
    topK
        The number of highest probability vocabulary tokens to keep for top-k-filtering
    topP
        If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are
        kept for generation
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty.
        See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once
    beamSize
        The Number of beams for beam search

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> audioAssembler = AudioAssembler() \\
    ...     .setInputCol("audio_content") \\
    ...     .setOutputCol("audio_assembler")
    >>> speechToText = WhisperForCTC.pretrained() \\
    ...     .setInputCols(["audio_assembler"]) \\
    ...     .setOutputCol("text")
    >>> pipeline = Pipeline().setStages([audioAssembler, speechToText])
    >>> processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
    >>> result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
    >>> result.select("text.result").show(truncate = False)
    +------------------------------------------------------------------------------------------+
    |result                                                                                    |
    +------------------------------------------------------------------------------------------+
    |[ Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.]|
    +------------------------------------------------------------------------------------------+
    """
    name = "WhisperForCTC"

    inputAnnotatorTypes = [AnnotatorType.AUDIO]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    language = Param(Params._dummy(), "language", "Optional parameter to set the language for the transcription.",
                     typeConverter=TypeConverters.toString)

    isMultilingual = Param(Params._dummy(), "isMultilingual", "Whether the model is multilingual.",
                           typeConverter=TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def getLanguage(self):
        """Gets the langauge for the transcription."""
        return self.getOrDefault(self.language)

    def getIsMultilingual(self):
        """Gets whether the model is multilingual."""
        return self.getOrDefault(self.isMultilingual)

    def setLanguage(self, value):
        """Sets the language for the audio, formatted to e.g. `<|en|>`. Check the model description for
        supported languages.

        Parameters
        ----------
        value : String
            Formatted language code
        """
        return self._call_java("setLanguage", value)

    def setTask(self, value):
        """Sets the formatted task for the audio. Either `<|translate|>` or `<|transcribe|>`.

        Only multilingual models can do translation.

        Parameters
        ----------
        value : String
            Formatted task
        """
        return self._call_java("setTask", value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC",
                 java_model=None):
        super(WhisperForCTC, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            minOutputLength=0,
            maxOutputLength=448,
            doSample=False,
            temperature=1.0,
            topK=1,
            topP=1.0,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            batchSize=2,
            beamSize=1,
            nReturnSequences=1,
            isMultilingual=True,
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
        WhisperForCTC
            The restored model
        """
        from sparknlp.internal import _WhisperForCTC
        jModel = _WhisperForCTC(folder, spark_session._jsparkSession)._java_obj
        return WhisperForCTC(java_model=jModel)

    @staticmethod
    def pretrained(name="asr_whisper_tiny_opt", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "asr_hubert_large_ls960"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        WhisperForCTC
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WhisperForCTC, name, lang, remote_loc)
