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

class JanusForMultiModal(AnnotatorModel,
                          HasBatchedAnnotateImage,
                          HasImageFeatureProperties,
                          HasEngine,
                          HasCandidateLabelsProperties,
                          HasRescaleFactor):
    """
    JanusForMultiModal can load Janus Vision models for visual question answering.
    The model consists of a vision encoder, a text encoder, and a text decoder.
    The vision encoder encodes the input image, the text encoder processes the input question
    alongside the image encoding, and the text decoder generates the answer to the question.

    Janus is a novel autoregressive framework that unifies multimodal understanding and generation.
    It decouples visual encoding into separate pathways while utilizing a single, unified transformer architecture for processing.
    This decoupling alleviates conflicts between the visual encoder’s roles in understanding and generation, enhancing the framework’s flexibility.

    Janus surpasses previous unified models and matches or exceeds the performance of task-specific models.
    It uses the DeepSeek-LLM-1.3b-base trained on approximately 500B text tokens.
    For multimodal understanding, it employs the SigLIP-L vision encoder supporting 384 x 384 image input,
    and for image generation, it uses a tokenizer with a downsample rate of 16.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> visualQAClassifier = JanusForMultiModal.pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("answer")

    The default model is `"janus_1_3b_int4"`, if no name is provided.
    For available pretrained models, refer to the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

    Models from the HuggingFace 🧧 Transformers library are also compatible with Spark NLP 🚀.
    To check compatibility and learn how to import them, see `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.
    For extended examples, refer to the `JanusForMultiModal Test Suite
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/JanusForMultiModalTest.scala>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize : int, optional
        Batch size. Larger values allow faster processing but require more memory,
        by default 2.
    configProtoBytes : bytes, optional
        ConfigProto from TensorFlow, serialized into a byte array.
    maxSentenceLength : int, optional
        Maximum sentence length to process, by default 50.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.sql.functions import lit
    >>> image_df = SparkSessionForTest.spark.read.format("image").load(path=images_path)
    >>> test_df = image_df.withColumn(
    ...     "text",
    ...     lit("User: <image_placeholder>Describe image in details\\n\\nAssistant:")
    ... )
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQAClassifier = JanusForMultiModal.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQAClassifier
    ... ])
    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("image_assembler.origin", "answer.result").show(truncate=False)
    +--------------------------------------+----------------------------------------------------------------------+
    |origin                                |result                                                                |
    +--------------------------------------+----------------------------------------------------------------------+
    |[file:///content/images/cat_image.jpg]|[The unusual aspect of this picture is the presence of two cats lying on a pink couch]|
    +--------------------------------------+----------------------------------------------------------------------+
    """



    name = "JanusForMultiModal"

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
    imageGenerateMode = Param(Params._dummy(), "imageGenerateMode",
                      "Image generation mode",
                      typeConverter=TypeConverters.toBoolean)
    numOfParallelImages = Param(Params._dummy(), "numOfParallelImages",
                    "Number of parallel images to Generate",
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

    def setImageGenerateMode(self, value):
        """Sets the image generation mode.
        Parameters
        ----------
        value : bool
            Image generation mode
        """
        return self._set(imageGenerateMode=value)

    def setNumOfParallelImages(self, value):
        """Sets the number of parallel images to generate.
        Parameters
        ----------
        value : int
            Number of parallel images to generate
        """
        return self._set(numOfParallelImages=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.JanusForMultiModal",
                 java_model=None):
        super(JanusForMultiModal, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=1,
            minOutputLength=0,
            maxOutputLength=50,
            doSample=False,
            temperature=1,
            topK=50,
            topP=1,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[],
            beamSize=1,
            imageGenerateMode=False,
            numOfParallelImages=1
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
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.internal import _JanusForMultiModalLoader
        jModel = _JanusForMultiModalLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return JanusForMultiModal(java_model=jModel)

    @staticmethod
    def pretrained(name="janus_1_3b_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.
        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "janus_1_3b_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.
        Returns
        -------
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(JanusForMultiModal, name, lang, remote_loc)