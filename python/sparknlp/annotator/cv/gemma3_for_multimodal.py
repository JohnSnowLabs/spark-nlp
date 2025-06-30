#  Copyright 2017-2024 John Snow Labs
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

from sparknlp.common import *

class Gemma3ForMultiModal(AnnotatorModel,
                 HasBatchedAnnotateImage,
                 HasImageFeatureProperties,
                 HasEngine,
                 HasGeneratorProperties):
    """Gemma3ForMultiModal can load Gemma 3 Vision models for visual question answering.
    The model consists of a vision encoder, a text encoder, a text decoder and a model merger.
    The vision encoder will encode the input image, the text encoder will encode the input text,
    the model merger will merge the image and text embeddings, and the text decoder will output the answer.

    Gemma 3 is a family of lightweight, state-of-the-art open models from Google, built from the same 
    research and technology used to create the Gemini models. It features:
    - Large 128K context window
    - Multilingual support in over 140 languages
    - Multimodal capabilities handling both text and image inputs
    - Optimized for deployment on limited resources (laptops, desktops, cloud)

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> visualQA = Gemma3ForMultiModal.pretrained() \
    ...     .setInputCols(["image_assembler"]) \
    ...     .setOutputCol("answer")

    The default model is ``"gemma3_4b_it_int4"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 1
    minOutputLength
        Minimum length of the sequence to be generated, by default 0
    maxOutputLength
        Maximum length of output text, by default 20
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default False
    temperature
        The value used to module the next token probabilities, by default 0.6
    topK
        The number of highest probability vocabulary tokens to keep for top-k-filtering, by default -1
    topP
        If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation, by default 0.9
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty, by default 1.0
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by default 3
    beamSize
        The Number of beams for beam search, by default 1
    maxInputLength
        Maximum length of input text, by default 4096

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.sql.functions import lit
    >>> imageDF = spark.read.format("image").load(images_path)
    >>> testDF = imageDF.withColumn("text", lit("<bos><start_of_turn>user\\nYou are a helpful assistant.\\n\\n<start_of_image>Describe this image in detail.<end_of_turn>\\n<start_of_turn>model\\n"))
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQA = Gemma3ForMultiModal.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQA
    ... ])
    >>> result = pipeline.fit(testDF).transform(testDF)
    >>> result.select("image_assembler.origin", "answer.result").show(truncate=False)
    """

    name = "Gemma3ForMultiModal"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
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
                     "The Number of beams for beam search.",
                     typeConverter=TypeConverters.toInt)

    maxInputLength = Param(Params._dummy(), "maxInputLength", "Maximum length of input text",
                            typeConverter=TypeConverters.toInt)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence length that the annotator will process, by
        default 50.

        Parameters
        ----------
        value : int
            Maximum sentence length that the annotator will process
        """
        return self._set(maxSentenceLength=value)

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
        """Sets the number of beam size for beam search, by default `4`.

        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.Gemma3ForMultiModal",
                 java_model=None):
        super(Gemma3ForMultiModal, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=1,
            minOutputLength=0,
            maxOutputLength=20,
            doSample=False,
            temperature=0.6,
            topK=-1,
            topP=0.9,
            repetitionPenalty=1.0,
            noRepeatNgramSize=3,
            ignoreTokenIds=[],
            beamSize=1,
            maxInputLength=4096,
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
        Gemma3ForMultiModal
            The restored model
        """
        from sparknlp.internal import _Gemma3ForMultiModalLoader
        jModel = _Gemma3ForMultiModalLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return Gemma3ForMultiModal(java_model=jModel)

    @staticmethod
    def pretrained(name="gemma3_4b_it_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "gemma3_4b_it_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        Gemma3ForMultiModal
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(Gemma3ForMultiModal, name, lang, remote_loc) 