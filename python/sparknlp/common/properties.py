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
"""Contains classes for Annotator properties."""

from pyspark.ml.param import Param, Params, TypeConverters


class HasBatchedAnnotate:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasCaseSensitiveProperties:
    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in tokens for embeddings matching",
                          typeConverter=TypeConverters.toBoolean)

    def setCaseSensitive(self, value):
        """Sets whether to ignore case in tokens for embeddings matching.

        Parameters
        ----------
        value : bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self._set(caseSensitive=value)

    def getCaseSensitive(self):
        """Gets whether to ignore case in tokens for embeddings matching.

        Returns
        -------
        bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self.getOrDefault(self.caseSensitive)


class HasClassifierActivationProperties:
    activation = Param(Params._dummy(),
                       "activation",
                       "Whether to calculate logits via Softmax or Sigmoid. Default is Softmax",
                       typeConverter=TypeConverters.toString)

    multilabel = Param(Params._dummy(),
                       "multilabel",
                       "Whether to calculate logits via Multiclass(softmax) or Multilabel(sigmoid). Default is False i.e. Multiclass",
                       typeConverter=TypeConverters.toBoolean)

    threshold = Param(Params._dummy(),
                      "threshold",
                      "Choose the threshold to determine which logits are considered to be positive or negative",
                      typeConverter=TypeConverters.toFloat)

    def setActivation(self, value):
        """Sets whether to calculate logits via Softmax or Sigmoid. Default is Softmax

        Parameters
        ----------
        value : str
            Whether to calculate logits via Softmax or Sigmoid. Default is Softmax
        """
        return self._set(activation=value)

    def getActivation(self):
        """Gets whether to calculate logits via Softmax or Sigmoid. Default is Softmax

        Returns
        -------
        str
            Whether to calculate logits via Softmax or Sigmoid. Default is Softmax
        """
        return self.getOrDefault(self.activation)

    def setMultilabel(self, value):
        """Set whether or not the result should be multi-class (the sum of all probabilities is 1.0) or
         multi-label (each label has a probability between 0.0 to 1.0).
         Default is False i.e. multi-class

        Parameters
        ----------
        value : bool
            Whether or not the result should be multi-class (the sum of all probabilities is 1.0) or
            multi-label (each label has a probability between 0.0 to 1.0).
            Default is False i.e. multi-class
        """
        return self._set(multilabel=value)

    def getMultilabel(self):
        """Gets whether or not the result should be multi-class (the sum of all probabilities is 1.0) or
         multi-label (each label has a probability between 0.0 to 1.0).
         Default is False i.e. multi-class

        Parameters
        ----------
        value : bool
            Whether or not the result should be multi-class (the sum of all probabilities is 1.0) or
            multi-label (each label has a probability between 0.0 to 1.0).
            Default is False i.e. multi-class
        """
        return self.getOrDefault(self.multilabel)

    def setThreshold(self, value):
        """Set the threshold to determine which logits are considered to be positive or negative.
         (Default: `0.5`). The value should be between 0.0 and 1.0. Changing the threshold value
         will affect the resulting labels and can be used to adjust the balance between precision and
         recall in the classification process.

        Parameters
        ----------
        value : float
            The threshold to determine which logits are considered to be positive or negative.
            (Default: `0.5`). The value should be between 0.0 and 1.0. Changing the threshold value
            will affect the resulting labels and can be used to adjust the balance between precision and
            recall in the classification process.
        """
        return self._set(threshold=value)


class HasEmbeddingsProperties(Params):
    dimension = Param(Params._dummy(),
                      "dimension",
                      "Number of embedding dimensions",
                      typeConverter=TypeConverters.toInt)

    def setDimension(self, value):
        """Sets embeddings dimension.

        Parameters
        ----------
        value : int
            Embeddings dimension
        """
        return self._set(dimension=value)

    def getDimension(self):
        """Gets embeddings dimension."""
        return self.getOrDefault(self.dimension)


class HasEnableCachingProperties:
    enableCaching = Param(Params._dummy(),
                          "enableCaching",
                          "Whether to enable caching DataFrames or RDDs during the training",
                          typeConverter=TypeConverters.toBoolean)

    def setEnableCaching(self, value):
        """Sets whether to enable caching DataFrames or RDDs during the training

        Parameters
        ----------
        value : bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self._set(enableCaching=value)

    def getEnableCaching(self):
        """Gets whether to enable caching DataFrames or RDDs during the training

        Returns
        -------
        bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self.getOrDefault(self.enableCaching)


class HasBatchedAnnotateImage:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasImageFeatureProperties:
    doResize = Param(Params._dummy(), "doResize", "Whether to resize the input to a certain size",
                     TypeConverters.toBoolean)

    doNormalize = Param(Params._dummy(), "doNormalize",
                        "Whether to normalize the input with mean and standard deviation",
                        TypeConverters.toBoolean)

    featureExtractorType = Param(Params._dummy(), "featureExtractorType",
                                 "Name of model's architecture for feature extraction",
                                 TypeConverters.toString)

    imageMean = Param(Params._dummy(), "imageMean",
                      "The sequence of means for each channel, to be used when normalizing images",
                      TypeConverters.toListFloat)

    imageStd = Param(Params._dummy(), "imageStd",
                     "The sequence of standard deviations for each channel, to be used when normalizing images",
                     TypeConverters.toListFloat)

    resample = Param(Params._dummy(), "resample",
                     "An optional resampling filter. This can be one of PIL.Image.NEAREST, PIL.Image.BILINEAR or "
                     "PIL.Image.BICUBIC. Only has an effect if do_resize is set to True.",
                     TypeConverters.toInt)

    size = Param(Params._dummy(), "size",
                 "Resize the input to the given size. If a tuple is provided, it should be (width, height). If only "
                 "an integer is provided, then the input will be resized to (size, size). Only has an effect if "
                 "do_resize is set to True.",
                 TypeConverters.toInt)

    def setDoResize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to resize the input to a certain size
        """
        return self._set(doResize=value)

    def setDoNormalize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to normalize the input with mean and standard deviation
        """
        return self._set(doNormalize=value)

    def setFeatureExtractorType(self, value):
        """

        Parameters
        ----------
        value : str
            Name of model's architecture for feature extraction
        """
        return self._set(featureExtractorType=value)

    def setImageStd(self, value):
        """

        Parameters
        ----------
        value : List[float]
            The sequence of standard deviations for each channel, to be used when normalizing images
        """
        return self._set(imageStd=value)

    def setImageMean(self, value):
        """

        Parameters
        ----------
        value : List[float]
            The sequence of means for each channel, to be used when normalizing images
        """
        return self._set(imageMean=value)

    def setResample(self, value):
        """

        Parameters
        ----------
        value : int
            Resampling filter for resizing. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BILINEAR` or
            `PIL.Image.BICUBIC`. Only has an effect if `do_resize` is set to `True`.
        """
        return self._set(resample=value)

    def setSize(self, value):
        """

        Parameters
        ----------
        value : int
            Resize the input to the given size. If a tuple is provided, it should be (width, height).
        """
        return self._set(size=value)


class HasRescaleFactor:
    doRescale = Param(Params._dummy(), "doRescale",
                      "Whether to rescale the image values by rescaleFactor.",
                      TypeConverters.toBoolean)

    rescaleFactor = Param(Params._dummy(), "rescaleFactor",
                          "Factor to scale the image values",
                          TypeConverters.toFloat)

    def setDoRescale(self, value):
        """Sets Whether to rescale the image values by rescaleFactor, by default `True`.

        Parameters
        ----------
        value : Boolean
            Whether to rescale the image values by rescaleFactor.
        """
        return self._set(doRescale=value)

    def setRescaleFactor(self, value):
        """Sets Factor to scale the image values, by default `1/255.0`.

        Parameters
        ----------
        value : Boolean
            Whether to rescale the image values by rescaleFactor.
        """
        return self._set(rescaleFactor=value)


class HasBatchedAnnotateAudio:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasAudioFeatureProperties:
    doNormalize = Param(Params._dummy(), "doNormalize",
                        "Whether to normalize the input",
                        TypeConverters.toBoolean)

    returnAttentionMask = Param(Params._dummy(), "returnAttentionMask", "",
                                TypeConverters.toBoolean)

    paddingSide = Param(Params._dummy(), "paddingSide",
                        "",
                        TypeConverters.toString)

    featureSize = Param(Params._dummy(), "featureSize",
                        "",
                        TypeConverters.toInt)

    samplingRate = Param(Params._dummy(), "samplingRate",
                         "",
                         TypeConverters.toInt)

    paddingValue = Param(Params._dummy(), "paddingValue",
                         "",
                         TypeConverters.toFloat)

    def setDoNormalize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to normalize the input with mean and standard deviation
        """
        return self._set(doNormalize=value)

    def setReturnAttentionMask(self, value):
        """

        Parameters
        ----------
        value : boolean
        """
        return self._set(returnAttentionMask=value)

    def setPaddingSide(self, value):
        """

        Parameters
        ----------
        value : str

        """
        return self._set(paddingSide=value)

    def setFeatureSize(self, value):
        """

        Parameters
        ----------
        value : int

        """
        return self._set(featureSize=value)

    def setSamplingRate(self, value):
        """

        Parameters
        ----------
        value : Int
        """
        return self._set(samplingRate=value)

    def setPaddingValue(self, value):
        """

        Parameters
        ----------
        value : float
        """
        return self._set(paddingValue=value)


class HasEngine:
    engine = Param(Params._dummy(), "engine",
                   "Deep Learning engine used for this model",
                   typeConverter=TypeConverters.toString)

    def getEngine(self):
        """

        Returns
        -------
        str
           Deep Learning engine used for this model"
        """
        return self.getOrDefault(self.engine)


class HasCandidateLabelsProperties:
    candidateLabels = Param(Params._dummy(), "candidateLabels",
                            "Deep Learning engine used for this model",
                            typeConverter=TypeConverters.toListString)

    contradictionIdParam = Param(Params._dummy(), "contradictionIdParam",
                                 "contradictionIdParam",
                                 typeConverter=TypeConverters.toInt)

    entailmentIdParam = Param(Params._dummy(), "entailmentIdParam",
                              "contradictionIdParam",
                              typeConverter=TypeConverters.toInt)

    def setCandidateLabels(self, v):
        """Sets candidateLabels.

        Parameters
        ----------
        v : list[string]
            candidateLabels
        """
        return self._set(candidateLabels=v)

    def setContradictionIdParam(self, v):
        """Sets contradictionIdParam.

        Parameters
        ----------
        v : int
            contradictionIdParam
        """
        return self._set(contradictionIdParam=v)

    def setEntailmentIdParam(self, v):
        """Sets entailmentIdParam.

        Parameters
        ----------
        v : int
            entailmentIdParam
        """
        return self._set(entailmentIdParam=v)


class HasMaxSentenceLengthLimit:
    # Default Value, can be overridden
    max_length_limit = 512

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Note that a maximum limit exists depending on the model. If you are working with long single
        sequences, consider splitting up the input first with another annotator e.g. SentenceDetector.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        if value > self.max_length_limit:
            raise ValueError(
                f"{self.__class__.__name__} models do not support token sequences longer than {self.max_length_limit}.\n"
                f"Consider splitting up the input first with another annotator e.g. SentenceDetector.")
        return self._set(maxSentenceLength=value)

    def getMaxSentenceLength(self):
        """Gets max sentence of the model.

        Returns
        -------
        int
            Max sentence length to process
        """
        return self.getOrDefault("maxSentenceLength")


class HasLongMaxSentenceLengthLimit(HasMaxSentenceLengthLimit):
    max_length_limit = 4096


class HasGeneratorProperties:
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

    beamSize = Param(Params._dummy(), "beamSize",
                     "The Number of beams for beam search.",
                     typeConverter=TypeConverters.toInt)

    nReturnSequences = Param(Params._dummy(),
                             "nReturnSequences",
                             "The number of sequences to return from the beam search.",
                             typeConverter=TypeConverters.toInt)


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
    """Sets the number of beam size for beam search.

    Parameters
    ----------
    value : int
        Number of beam size for beam search
    """
    return self._set(beamSize=value)


def setNReturnSequences(self, value):
    """Sets the number of sequences to return from the beam search.

    Parameters
    ----------
    value : int
        Number of sequences to return
    """
    return self._set(nReturnSequences=value)
