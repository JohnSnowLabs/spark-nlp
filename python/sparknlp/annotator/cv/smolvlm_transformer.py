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

class SmolVLMTransformer(AnnotatorModel,
                          HasBatchedAnnotateImage,
                          HasImageFeatureProperties,
                          HasEngine,
                          HasCandidateLabelsProperties,
                          HasRescaleFactor):
    """
    SmolVLMTransformer can load SmolVLM models for visual question answering. The model
    consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder
    will encode the input image, the text encoder will encode the input question together with the
    encoding of the image, and the text decoder will output the answer to the question.

    SmolVLM is a compact open multimodal model that accepts arbitrary sequences of image and text
    inputs to produce text outputs. Designed for efficiency, SmolVLM can answer questions about images,
    describe visual content, create stories grounded on multiple images, or function as a pure language
    model without visual inputs. Its lightweight architecture makes it suitable for on-device applications
    while maintaining strong performance on multimodal tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:
    >>> visualQA = SmolVLMTransformer.pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("answer")

    The default model is `"smolvlm_instruct_int4"`, if no name is provided.
    For available pretrained models, refer to the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

    Models from the HuggingFace 🧧 Transformers library are also compatible with Spark NLP 🚀.
    To check compatibility and learn how to import them, see `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.
    For extended examples, refer to the `SmolVLMTransformer Test Suite
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/SmolVLMTransformerTest.scala>`_.

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
    configProtoBytes : bytes, optional
        ConfigProto from TensorFlow, serialized into a byte array.
    maxSentenceLength : int, optional
        Maximum sentence length to process, by default 20.
    doImageSplitting : bool, optional
        Whether to split the image, by default True.
    imageToken : int, optional
        Token ID for image embeddings, by default 49153.
    numVisionTokens : int, optional
        Number of vision tokens, by default 81.
    maxImageSize : int, optional
        Maximum image size for the model, by default 384.
    patchSize : int, optional
        Patch size for the model, by default 14.
    paddingConstant : int, optional
        Padding constant for the model, by default 0.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.sql.functions import lit
    >>> imageDF = spark.read.format("image").load(path=images_path)
    >>> testDF = imageDF.withColumn(
    ...     "text",
    ...     lit("<|im_start|>User:<image>Can you describe the image?<end_of_utterance>\\nAssistant:")
    ... )
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQAClassifier = SmolVLMTransformer.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQAClassifier
    ... ])
    >>> result = pipeline.fit(testDF).transform(testDF)
    >>> result.select("image_assembler.origin", "answer.result").show(truncate=False)
    +--------------------------------------+----------------------------------------------------------------------+
    |origin                                |result                                                                |
    +--------------------------------------+----------------------------------------------------------------------+
    |[file:///content/images/cat_image.jpg]|[The unusual aspect of this picture is the presence of two cats lying on a pink couch]|
    +--------------------------------------+----------------------------------------------------------------------+
    """

    name = "SmolVLMTransformer"

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
                     
    stopTokenIds = Param(Params._dummy(), "stopTokenIds",
                       "Stop tokens to terminate the generation",
                       typeConverter=TypeConverters.toListInt)
                       
    imageToken = Param(Params._dummy(), "imageToken",
                     "Token id for image embeddings",
                     typeConverter=TypeConverters.toInt)
                     
    numVisionTokens = Param(Params._dummy(), "numVisionTokens",
                         "Number of vision tokens",
                         typeConverter=TypeConverters.toInt)
                         
    maxImageSize = Param(Params._dummy(), "maxImageSize",
                      "Maximum image size for the model",
                      typeConverter=TypeConverters.toInt)
                      
    patchSize = Param(Params._dummy(), "patchSize",
                     "Patch size for the model",
                     typeConverter=TypeConverters.toInt)
                     
    paddingConstant = Param(Params._dummy(), "paddingConstant",
                          "Padding constant for the model",
                          typeConverter=TypeConverters.toInt)
                          
    doImageSplitting = Param(Params._dummy(), "doImageSplitting",
                          "Whether to split the image",
                          typeConverter=TypeConverters.toBoolean)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence length that the annotator will process, by
        default 20.
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
        
    def setStopTokenIds(self, value):
        """Stop tokens to terminate the generation.
        Parameters
        ----------
        value : List[int]
            The tokens that terminate generation
        """
        return self._set(stopTokenIds=value)

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
        """Sets the number of beam size for beam search, by default `1`.
        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)
        
    def setImageToken(self, value):
        """Sets the token ID for image embeddings.
        Parameters
        ----------
        value : int
            Token ID for image embeddings
        """
        return self._set(imageToken=value)
        
    def setNumVisionTokens(self, value):
        """Sets the number of vision tokens.
        Parameters
        ----------
        value : int
            Number of vision tokens
        """
        return self._set(numVisionTokens=value)
        
    def setMaxImageSize(self, value):
        """Sets the maximum image size for the model.
        Parameters
        ----------
        value : int
            Maximum image size
        """
        return self._set(maxImageSize=value)
        
    def setPatchSize(self, value):
        """Sets the patch size for the model.
        Parameters
        ----------
        value : int
            Patch size
        """
        return self._set(patchSize=value)
        
    def setPaddingConstant(self, value):
        """Sets the padding constant for the model.
        Parameters
        ----------
        value : int
            Padding constant
        """
        return self._set(paddingConstant=value)
        
    def setDoImageSplitting(self, value):
        """Sets whether to split the image.
        Parameters
        ----------
        value : bool
            Whether to split the image
        """
        return self._set(doImageSplitting=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.SmolVLMTransformer",
                 java_model=None):
        super(SmolVLMTransformer, self).__init__(
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
            stopTokenIds=[49154],
            imageToken=49153,
            numVisionTokens=81,
            maxImageSize=384,
            patchSize=14,
            paddingConstant=0,
            doImageSplitting=True
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
        use_openvino : bool, optional
            Whether to use OpenVINO for inference, by default False
        Returns
        -------
        SmolVLMTransformer
            The restored model
        """
        from sparknlp.internal import _SmolVLMTransformerLoader
        jModel = _SmolVLMTransformerLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return SmolVLMTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="smolvlm_instruct_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.
        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "smolvlm_instruct_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.
        Returns
        -------
        SmolVLMTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SmolVLMTransformer, name, lang, remote_loc) 