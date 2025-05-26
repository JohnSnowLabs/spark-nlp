from sparknlp.common import *

class InternVLForMultiModal(AnnotatorModel,
                          HasBatchedAnnotateImage,
                          HasImageFeatureProperties,
                          HasEngine,
                          HasGeneratorProperties):
    """
    InternVLForMultiModal can load InternVL Vision models for visual question answering.
    The model consists of a vision encoder, a text encoder, a text decoder and a model merger.
    The vision encoder will encode the input image, the text encoder will encode the input text,
    the model merger will merge the image and text embeddings, and the text decoder will output the answer.

    InternVL 2.5 is an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0,
    maintaining its core model architecture while introducing significant enhancements in training and testing
    strategies as well as data quality. Key features include:
    - Large context window support
    - Multilingual support
    - Multimodal capabilities handling both text and image inputs
    - Optimized for deployment with int4 quantization

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:
    >>> visualQA = InternVLForMultiModal.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")

    The default model is `"internvl2_5_1b_int4"`, if no name is provided.
    For available pretrained models, refer to the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize : int, optional
        Batch size. Larger values allow faster processing but require more memory,
        by default 1.
    maxSentenceLength : int, optional
        Maximum sentence length to process, by default 4096.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.sql.functions import lit
    >>> image_df = spark.read.format("image").load(path=images_path)
    >>> test_df = image_df.withColumn(
    ...     "text",
    ...     lit("<|im_start|><image>\\nDescribe this image in detail.<|im_end|><|im_start|>assistant\\n")
    ... )
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQA = InternVLForMultiModal.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQA
    ... ])

    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("image_assembler.origin", "answer.result").show(truncate=False)
    """

    name = "InternVLForMultiModal"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

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

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence length that the annotator will process, by
        default 4096.
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
        """Sets the number of beam size for beam search, by default `1`.
        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.InternVLForMultiModal",
                 java_model=None):
        super(InternVLForMultiModal, self).__init__(
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
            beamSize=1
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
        InternVLForMultiModal
            The restored model
        """
        from sparknlp.internal import _InternVLForMultiModalLoader
        jModel = _InternVLForMultiModalLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return InternVLForMultiModal(java_model=jModel)

    @staticmethod
    def pretrained(name="internvl2_5_1b_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.
        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "internvl2_5_1b_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.
        Returns
        -------
        InternVLForMultiModal
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(InternVLForMultiModal, name, lang, remote_loc) 
