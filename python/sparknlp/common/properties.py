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
from typing import List, Dict

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
        return self.getOrDefault(self.batchSize)


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


class HasClsTokenProperties:
    useCLSToken = Param(Params._dummy(),
                        "useCLSToken",
                        "Whether to use CLS token for pooling (true) or attention-based average pooling (false)",
                        typeConverter=TypeConverters.toBoolean)

    def setUseCLSToken(self, value):
        """Sets whether to ignore case in tokens for embeddings matching.

        Parameters
        ----------
        value : bool
            Whether to use CLS token for pooling (true) or attention-based average pooling (false)
        """
        return self._set(useCLSToken=value)

    def getUseCLSToken(self):
        """Gets whether to use CLS token for pooling (true) or attention-based average pooling (false)

        Returns
        -------
        bool
            Whether to use CLS token for pooling (true) or attention-based average pooling (false)
        """
        return self.getOrDefault(self.useCLSToken)


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
        return self.getOrDefault(self.batchSize)


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
        return self.getOrDefault(self.batchSize)


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


class HasLlamaCppProperties:
    # -------- MODEl PARAMETERS --------
    nThreads = Param(Params._dummy(), "nThreads", "Set the number of threads to use during generation",
                     typeConverter=TypeConverters.toInt)
    # nThreadsDraft = Param(Params._dummy(), "nThreadsDraft", "Set the number of threads to use during draft generation",
    #                       typeConverter=TypeConverters.toInt)
    nThreadsBatch = Param(Params._dummy(), "nThreadsBatch",
                          "Set the number of threads to use during batch and prompt processing",
                          typeConverter=TypeConverters.toInt)
    # nThreadsBatchDraft = Param(Params._dummy(), "nThreadsBatchDraft",
    #                            "Set the number of threads to use during batch and prompt processing",
    #                            typeConverter=TypeConverters.toInt)
    nCtx = Param(Params._dummy(), "nCtx", "Set the size of the prompt context", typeConverter=TypeConverters.toInt)
    nBatch = Param(Params._dummy(), "nBatch",
                   "Set the logical batch size for prompt processing (must be >=32 to use BLAS)",
                   typeConverter=TypeConverters.toInt)
    nUbatch = Param(Params._dummy(), "nUbatch",
                    "Set the physical batch size for prompt processing (must be >=32 to use BLAS)",
                    typeConverter=TypeConverters.toInt)
    nDraft = Param(Params._dummy(), "nDraft", "Set the number of tokens to draft for speculative decoding",
                   typeConverter=TypeConverters.toInt)
    # nChunks = Param(Params._dummy(), "nChunks", "Set the maximal number of chunks to process",
    #                 typeConverter=TypeConverters.toInt)
    # nSequences = Param(Params._dummy(), "nSequences", "Set the number of sequences to decode",
    #                    typeConverter=TypeConverters.toInt)
    # pSplit = Param(Params._dummy(), "pSplit", "Set the speculative decoding split probability",
    #                typeConverter=TypeConverters.toFloat)
    nGpuLayers = Param(Params._dummy(), "nGpuLayers", "Set the number of layers to store in VRAM (-1 - use default)",
                       typeConverter=TypeConverters.toInt)
    nGpuLayersDraft = Param(Params._dummy(), "nGpuLayersDraft",
                            "Set the number of layers to store in VRAM for the draft model (-1 - use default)",
                            typeConverter=TypeConverters.toInt)
    # Set how to split the model across GPUs
    #
    #   - NONE: No GPU split
    #   - LAYER: Split the model across GPUs by layer
    #   - ROW: Split the model across GPUs by rows
    gpuSplitMode = Param(Params._dummy(), "gpuSplitMode", "Set how to split the model across GPUs",
                         typeConverter=TypeConverters.toString)
    mainGpu = Param(Params._dummy(), "mainGpu", "Set the main GPU that is used for scratch and small tensors.",
                    typeConverter=TypeConverters.toInt)
    # tensorSplit = Param(Params._dummy(), "tensorSplit", "Set how split tensors should be distributed across GPUs",
    #                     typeConverter=TypeConverters.toListFloat)
    # grpAttnN = Param(Params._dummy(), "grpAttnN", "Set the group-attention factor", typeConverter=TypeConverters.toInt)
    # grpAttnW = Param(Params._dummy(), "grpAttnW", "Set the group-attention width", typeConverter=TypeConverters.toInt)
    ropeFreqBase = Param(Params._dummy(), "ropeFreqBase", "Set the RoPE base frequency, used by NTK-aware scaling",
                         typeConverter=TypeConverters.toFloat)
    ropeFreqScale = Param(Params._dummy(), "ropeFreqScale",
                          "Set the RoPE frequency scaling factor, expands context by a factor of 1/N",
                          typeConverter=TypeConverters.toFloat)
    yarnExtFactor = Param(Params._dummy(), "yarnExtFactor", "Set the YaRN extrapolation mix factor",
                          typeConverter=TypeConverters.toFloat)
    yarnAttnFactor = Param(Params._dummy(), "yarnAttnFactor", "Set the YaRN scale sqrt(t) or attention magnitude",
                           typeConverter=TypeConverters.toFloat)
    yarnBetaFast = Param(Params._dummy(), "yarnBetaFast", "Set the YaRN low correction dim or beta",
                         typeConverter=TypeConverters.toFloat)
    yarnBetaSlow = Param(Params._dummy(), "yarnBetaSlow", "Set the YaRN high correction dim or alpha",
                         typeConverter=TypeConverters.toFloat)
    yarnOrigCtx = Param(Params._dummy(), "yarnOrigCtx", "Set the YaRN original context size of model",
                        typeConverter=TypeConverters.toInt)
    defragmentationThreshold = Param(Params._dummy(), "defragmentationThreshold",
                                     "Set the KV cache defragmentation threshold", typeConverter=TypeConverters.toFloat)
    # Set optimization strategies that help on some NUMA systems (if available)
    #
    # Available Strategies:
    #
    #   - DISABLED: No NUMA optimizations
    #   - DISTRIBUTE: Spread execution evenly over all
    #   - ISOLATE: Only spawn threads on CPUs on the node that execution started on
    #   - NUMA_CTL: Use the CPU map provided by numactl
    #   - MIRROR: Mirrors the model across NUMA nodes
    numaStrategy = Param(Params._dummy(), "numaStrategy",
                         "Set optimization strategies that help on some NUMA systems (if available)",
                         typeConverter=TypeConverters.toString)
    # Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    #
    #   - NONE: Don't use any scaling
    #   - LINEAR: Linear scaling
    #   - YARN: YaRN RoPE scaling
    ropeScalingType = Param(Params._dummy(), "ropeScalingType",
                            "Set the RoPE frequency scaling method, defaults to linear unless specified by the model",
                            typeConverter=TypeConverters.toString)
    # Set the pooling type for embeddings, use model default if unspecified
    #
    #   - MEAN: Mean Pooling
    #   - CLS: CLS Pooling
    #   - LAST: Last token pooling
    #   - RANK: For reranked models
    poolingType = Param(Params._dummy(), "poolingType",
                        "Set the pooling type for embeddings, use model default if unspecified",
                        typeConverter=TypeConverters.toString)
    modelDraft = Param(Params._dummy(), "modelDraft", "Set the draft model for speculative decoding",
                       typeConverter=TypeConverters.toString)
    modelAlias = Param(Params._dummy(), "modelAlias", "Set a model alias", typeConverter=TypeConverters.toString)
    # lookupCacheStaticFilePath = Param(Params._dummy(), "lookupCacheStaticFilePath",
    #                                   "Set path to static lookup cache to use for lookup decoding (not updated by generation)",
    #                                   typeConverter=TypeConverters.toString)
    # lookupCacheDynamicFilePath = Param(Params._dummy(), "lookupCacheDynamicFilePath",
    #                                    "Set path to dynamic lookup cache to use for lookup decoding (updated by generation)",
    #                                    typeConverter=TypeConverters.toString)
    # loraAdapters = new StructFeature[Map[String, Float]](this, "loraAdapters")
    embedding = Param(Params._dummy(), "embedding", "Whether to load model with embedding support",
                      typeConverter=TypeConverters.toBoolean)
    flashAttention = Param(Params._dummy(), "flashAttention", "Whether to enable Flash Attention",
                           typeConverter=TypeConverters.toBoolean)
    # inputPrefixBos = Param(Params._dummy(), "inputPrefixBos",
    #                        "Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string",
    #                        typeConverter=TypeConverters.toBoolean)
    useMmap = Param(Params._dummy(), "useMmap",
                    "Whether to use memory-map model (faster load but may increase pageouts if not using mlock)",
                    typeConverter=TypeConverters.toBoolean)
    useMlock = Param(Params._dummy(), "useMlock",
                     "Whether to force the system to keep model in RAM rather than swapping or compressing",
                     typeConverter=TypeConverters.toBoolean)
    noKvOffload = Param(Params._dummy(), "noKvOffload", "Whether to disable KV offload",
                        typeConverter=TypeConverters.toBoolean)
    systemPrompt = Param(Params._dummy(), "systemPrompt", "Set a system prompt to use",
                         typeConverter=TypeConverters.toString)
    chatTemplate = Param(Params._dummy(), "chatTemplate", "The chat template to use",
                         typeConverter=TypeConverters.toString)
    logVerbosity = Param(Params._dummy(), "logVerbosity", "Set the log verbosity level",
                         typeConverter=TypeConverters.toInt)
    disableLog = Param(Params._dummy(), "disableLog", "Whether to disable logging",
                       typeConverter=TypeConverters.toBoolean)

    # -------- INFERENCE PARAMETERS --------
    inputPrefix = Param(Params._dummy(), "inputPrefix", "Set the prompt to start generation with",
                        typeConverter=TypeConverters.toString)
    inputSuffix = Param(Params._dummy(), "inputSuffix", "Set a suffix for infilling",
                        typeConverter=TypeConverters.toString)
    cachePrompt = Param(Params._dummy(), "cachePrompt", "Whether to remember the prompt to avoid reprocessing it",
                        typeConverter=TypeConverters.toBoolean)
    nPredict = Param(Params._dummy(), "nPredict", "Set the number of tokens to predict",
                     typeConverter=TypeConverters.toInt)
    topK = Param(Params._dummy(), "topK", "Set top-k sampling", typeConverter=TypeConverters.toInt)
    topP = Param(Params._dummy(), "topP", "Set top-p sampling", typeConverter=TypeConverters.toFloat)
    minP = Param(Params._dummy(), "minP", "Set min-p sampling", typeConverter=TypeConverters.toFloat)
    tfsZ = Param(Params._dummy(), "tfsZ", "Set tail free sampling, parameter z", typeConverter=TypeConverters.toFloat)
    typicalP = Param(Params._dummy(), "typicalP", "Set locally typical sampling, parameter p",
                     typeConverter=TypeConverters.toFloat)
    temperature = Param(Params._dummy(), "temperature", "Set the temperature", typeConverter=TypeConverters.toFloat)
    dynamicTemperatureRange = Param(Params._dummy(), "dynatempRange", "Set the dynamic temperature range",
                                    typeConverter=TypeConverters.toFloat)
    dynamicTemperatureExponent = Param(Params._dummy(), "dynatempExponent", "Set the dynamic temperature exponent",
                                       typeConverter=TypeConverters.toFloat)
    repeatLastN = Param(Params._dummy(), "repeatLastN", "Set the last n tokens to consider for penalties",
                        typeConverter=TypeConverters.toInt)
    repeatPenalty = Param(Params._dummy(), "repeatPenalty", "Set the penalty of repeated sequences of tokens",
                          typeConverter=TypeConverters.toFloat)
    frequencyPenalty = Param(Params._dummy(), "frequencyPenalty", "Set the repetition alpha frequency penalty",
                             typeConverter=TypeConverters.toFloat)
    presencePenalty = Param(Params._dummy(), "presencePenalty", "Set the repetition alpha presence penalty",
                            typeConverter=TypeConverters.toFloat)
    miroStat = Param(Params._dummy(), "miroStat", "Set MiroStat sampling strategies.",
                     typeConverter=TypeConverters.toString)
    miroStatTau = Param(Params._dummy(), "mirostatTau", "Set the MiroStat target entropy, parameter tau",
                        typeConverter=TypeConverters.toFloat)
    miroStatEta = Param(Params._dummy(), "mirostatEta", "Set the MiroStat learning rate, parameter eta",
                        typeConverter=TypeConverters.toFloat)
    penalizeNl = Param(Params._dummy(), "penalizeNl", "Whether to penalize newline tokens",
                       typeConverter=TypeConverters.toBoolean)
    nKeep = Param(Params._dummy(), "nKeep", "Set the number of tokens to keep from the initial prompt",
                  typeConverter=TypeConverters.toInt)
    seed = Param(Params._dummy(), "seed", "Set the RNG seed", typeConverter=TypeConverters.toInt)
    nProbs = Param(Params._dummy(), "nProbs", "Set the amount top tokens probabilities to output if greater than 0.",
                   typeConverter=TypeConverters.toInt)
    minKeep = Param(Params._dummy(), "minKeep",
                    "Set the amount of tokens the samplers should return at least (0 = disabled)",
                    typeConverter=TypeConverters.toInt)
    grammar = Param(Params._dummy(), "grammar", "Set BNF-like grammar to constrain generations",
                    typeConverter=TypeConverters.toString)
    penaltyPrompt = Param(Params._dummy(), "penaltyPrompt",
                          "Override which part of the prompt is penalized for repetition.",
                          typeConverter=TypeConverters.toString)
    ignoreEos = Param(Params._dummy(), "ignoreEos",
                      "Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)",
                      typeConverter=TypeConverters.toBoolean)
    disableTokenIds = Param(Params._dummy(), "disableTokenIds", "Set the token ids to disable in the completion",
                            typeConverter=TypeConverters.toListInt)
    stopStrings = Param(Params._dummy(), "stopStrings", "Set strings upon seeing which token generation is stopped",
                        typeConverter=TypeConverters.toListString)
    samplers = Param(Params._dummy(), "samplers", "Set which samplers to use for token generation in the given order",
                     typeConverter=TypeConverters.toListString)
    useChatTemplate = Param(Params._dummy(), "useChatTemplate",
                            "Set whether or not generate should apply a chat template",
                            typeConverter=TypeConverters.toBoolean)

    # -------- MODEL SETTERS --------
    def setNThreads(self, nThreads: int):
        """Set the number of threads to use during generation"""
        return self._set(nThreads=nThreads)

    # def setNThreadsDraft(self, nThreadsDraft: int):
    #     """Set the number of threads to use during draft generation"""
    #     return self._set(nThreadsDraft=nThreadsDraft)

    def setNThreadsBatch(self, nThreadsBatch: int):
        """Set the number of threads to use during batch and prompt processing"""
        return self._set(nThreadsBatch=nThreadsBatch)

    # def setNThreadsBatchDraft(self, nThreadsBatchDraft: int):
    #     """Set the number of threads to use during batch and prompt processing"""
    #     return self._set(nThreadsBatchDraft=nThreadsBatchDraft)

    def setNCtx(self, nCtx: int):
        """Set the size of the prompt context"""
        return self._set(nCtx=nCtx)

    def setNBatch(self, nBatch: int):
        """Set the logical batch size for prompt processing (must be >=32 to use BLAS)"""
        return self._set(nBatch=nBatch)

    def setNUbatch(self, nUbatch: int):
        """Set the physical batch size for prompt processing (must be >=32 to use BLAS)"""
        return self._set(nUbatch=nUbatch)

    def setNDraft(self, nDraft: int):
        """Set the number of tokens to draft for speculative decoding"""
        return self._set(nDraft=nDraft)

    # def setNChunks(self, nChunks: int):
    #     """Set the maximal number of chunks to process"""
    #     return self._set(nChunks=nChunks)

    # def setNSequences(self, nSequences: int):
    #     """Set the number of sequences to decode"""
    #     return self._set(nSequences=nSequences)

    # def setPSplit(self, pSplit: float):
    #     """Set the speculative decoding split probability"""
    #     return self._set(pSplit=pSplit)

    def setNGpuLayers(self, nGpuLayers: int):
        """Set the number of layers to store in VRAM (-1 - use default)"""
        return self._set(nGpuLayers=nGpuLayers)

    def setNGpuLayersDraft(self, nGpuLayersDraft: int):
        """Set the number of layers to store in VRAM for the draft model (-1 - use default)"""
        return self._set(nGpuLayersDraft=nGpuLayersDraft)

    def setGpuSplitMode(self, gpuSplitMode: str):
        """Set how to split the model across GPUs"""
        return self._set(gpuSplitMode=gpuSplitMode)

    def setMainGpu(self, mainGpu: int):
        """Set the main GPU that is used for scratch and small tensors."""
        return self._set(mainGpu=mainGpu)

    # def setTensorSplit(self, tensorSplit: List[float]):
    #     """Set how split tensors should be distributed across GPUs"""
    #     return self._set(tensorSplit=tensorSplit)

    # def setGrpAttnN(self, grpAttnN: int):
    #     """Set the group-attention factor"""
    #     return self._set(grpAttnN=grpAttnN)

    # def setGrpAttnW(self, grpAttnW: int):
    #     """Set the group-attention width"""
    #     return self._set(grpAttnW=grpAttnW)

    def setRopeFreqBase(self, ropeFreqBase: float):
        """Set the RoPE base frequency, used by NTK-aware scaling"""
        return self._set(ropeFreqBase=ropeFreqBase)

    def setRopeFreqScale(self, ropeFreqScale: float):
        """Set the RoPE frequency scaling factor, expands context by a factor of 1/N"""
        return self._set(ropeFreqScale=ropeFreqScale)

    def setYarnExtFactor(self, yarnExtFactor: float):
        """Set the YaRN extrapolation mix factor"""
        return self._set(yarnExtFactor=yarnExtFactor)

    def setYarnAttnFactor(self, yarnAttnFactor: float):
        """Set the YaRN scale sqrt(t) or attention magnitude"""
        return self._set(yarnAttnFactor=yarnAttnFactor)

    def setYarnBetaFast(self, yarnBetaFast: float):
        """Set the YaRN low correction dim or beta"""
        return self._set(yarnBetaFast=yarnBetaFast)

    def setYarnBetaSlow(self, yarnBetaSlow: float):
        """Set the YaRN high correction dim or alpha"""
        return self._set(yarnBetaSlow=yarnBetaSlow)

    def setYarnOrigCtx(self, yarnOrigCtx: int):
        """Set the YaRN original context size of model"""
        return self._set(yarnOrigCtx=yarnOrigCtx)

    def setDefragmentationThreshold(self, defragmentationThreshold: float):
        """Set the KV cache defragmentation threshold"""
        return self._set(defragmentationThreshold=defragmentationThreshold)

    def setNumaStrategy(self, numaStrategy: str):
        """Set optimization strategies that help on some NUMA systems (if available)

        Possible values:

        - DISABLED: No NUMA optimizations
        - DISTRIBUTE: spread execution evenly over all
        - ISOLATE: only spawn threads on CPUs on the node that execution started on
        - NUMA_CTL: use the CPU map provided by numactl
        - MIRROR: Mirrors the model across NUMA nodes
        """
        numaUpper = numaStrategy.upper()
        numaStrategies = ["DISABLED", "DISTRIBUTE", "ISOLATE", "NUMA_CTL", "MIRROR"]
        if numaUpper not in numaStrategies:
            raise ValueError(
                f"Invalid NUMA strategy: {numaUpper}. "
                + f"Valid values are: {numaStrategies}"
            )
        return self._set(numaStrategy=numaStrategy)

    def setRopeScalingType(self, ropeScalingType: str):
        """Set the RoPE frequency scaling method, defaults to linear unless specified by the model.

        Possible values:

        - NONE: Don't use any scaling
        - LINEAR: Linear scaling
        - YARN: YaRN RoPE scaling
        """
        ropeScalingTypeUpper = ropeScalingType.upper()
        ropeScalingTypes = ["NONE", "LINEAR", "YARN"]
        if ropeScalingTypeUpper not in ropeScalingTypes:
            raise ValueError(
                f"Invalid RoPE scaling type: {ropeScalingType}. "
                + f"Valid values are: {ropeScalingTypes}"
            )
        return self._set(ropeScalingType=ropeScalingTypeUpper)

    def setPoolingType(self, poolingType: str):
        """Set the pooling type for embeddings, use model default if unspecified

        Possible values:

        - MEAN: Mean Pooling
        - CLS: CLS Pooling
        - LAST: Last token pooling
        - RANK: For reranked models
        """
        poolingTypeUpper = poolingType.upper()
        poolingTypes = ["NONE", "MEAN", "CLS", "LAST", "RANK"]
        if poolingTypeUpper not in poolingTypes:
            raise ValueError(
                f"Invalid pooling type: {poolingType}. "
                + f"Valid values are: {poolingTypes}"
            )
        return self._set(poolingType=poolingType)

    def setModelDraft(self, modelDraft: str):
        """Set the draft model for speculative decoding"""
        return self._set(modelDraft=modelDraft)

    def setModelAlias(self, modelAlias: str):
        """Set a model alias"""
        return self._set(modelAlias=modelAlias)

    # def setLookupCacheStaticFilePath(self, lookupCacheStaticFilePath: str):
    #     """Set path to static lookup cache to use for lookup decoding (not updated by generation)"""
    #     return self._set(lookupCacheStaticFilePath=lookupCacheStaticFilePath)

    # def setLookupCacheDynamicFilePath(self, lookupCacheDynamicFilePath: str):
    #     """Set path to dynamic lookup cache to use for lookup decoding (updated by generation)"""
    #     return self._set(lookupCacheDynamicFilePath=lookupCacheDynamicFilePath)

    def setFlashAttention(self, flashAttention: bool):
        """Whether to enable Flash Attention"""
        return self._set(flashAttention=flashAttention)

    # def setInputPrefixBos(self, inputPrefixBos: bool):
    #     """Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string"""
    #     return self._set(inputPrefixBos=inputPrefixBos)

    def setUseMmap(self, useMmap: bool):
        """Whether to use memory-map model (faster load but may increase pageouts if not using mlock)"""
        return self._set(useMmap=useMmap)

    def setUseMlock(self, useMlock: bool):
        """Whether to force the system to keep model in RAM rather than swapping or compressing"""
        return self._set(useMlock=useMlock)

    def setNoKvOffload(self, noKvOffload: bool):
        """Whether to disable KV offload"""
        return self._set(noKvOffload=noKvOffload)

    def setSystemPrompt(self, systemPrompt: str):
        """Set a system prompt to use"""
        return self._set(systemPrompt=systemPrompt)

    def setChatTemplate(self, chatTemplate: str):
        """The chat template to use"""
        return self._set(chatTemplate=chatTemplate)

    # -------- INFERENCE SETTERS --------
    def setInputPrefix(self, inputPrefix: str):
        """Set the prompt to start generation with"""
        return self._set(inputPrefix=inputPrefix)

    def setInputSuffix(self, inputSuffix: str):
        """Set a suffix for infilling"""
        return self._set(inputSuffix=inputSuffix)

    def setCachePrompt(self, cachePrompt: bool):
        """Whether to remember the prompt to avoid reprocessing it"""
        return self._set(cachePrompt=cachePrompt)

    def setNPredict(self, nPredict: int):
        """Set the number of tokens to predict"""
        return self._set(nPredict=nPredict)

    def setTopK(self, topK: int):
        """Set top-k sampling"""
        return self._set(topK=topK)

    def setTopP(self, topP: float):
        """Set top-p sampling"""
        return self._set(topP=topP)

    def setMinP(self, minP: float):
        """Set min-p sampling"""
        return self._set(minP=minP)

    def setTfsZ(self, tfsZ: float):
        """Set tail free sampling, parameter z"""
        return self._set(tfsZ=tfsZ)

    def setTypicalP(self, typicalP: float):
        """Set locally typical sampling, parameter p"""
        return self._set(typicalP=typicalP)

    def setTemperature(self, temperature: float):
        """Set the temperature"""
        return self._set(temperature=temperature)

    def setDynamicTemperatureRange(self, dynamicTemperatureRange: float):
        """Set the dynamic temperature range"""
        return self._set(dynamicTemperatureRange=dynamicTemperatureRange)

    def setDynamicTemperatureExponent(self, dynamicTemperatureExponent: float):
        """Set the dynamic temperature exponent"""
        return self._set(dynamicTemperatureExponent=dynamicTemperatureExponent)

    def setRepeatLastN(self, repeatLastN: int):
        """Set the last n tokens to consider for penalties"""
        return self._set(repeatLastN=repeatLastN)

    def setRepeatPenalty(self, repeatPenalty: float):
        """Set the penalty of repeated sequences of tokens"""
        return self._set(repeatPenalty=repeatPenalty)

    def setFrequencyPenalty(self, frequencyPenalty: float):
        """Set the repetition alpha frequency penalty"""
        return self._set(frequencyPenalty=frequencyPenalty)

    def setPresencePenalty(self, presencePenalty: float):
        """Set the repetition alpha presence penalty"""
        return self._set(presencePenalty=presencePenalty)

    def setMiroStat(self, miroStat: str):
        """Set MiroStat sampling strategies."""
        return self._set(miroStat=miroStat)

    def setMiroStatTau(self, miroStatTau: float):
        """Set the MiroStat target entropy, parameter tau"""
        return self._set(miroStatTau=miroStatTau)

    def setMiroStatEta(self, miroStatEta: float):
        """Set the MiroStat learning rate, parameter eta"""
        return self._set(miroStatEta=miroStatEta)

    def setPenalizeNl(self, penalizeNl: bool):
        """Whether to penalize newline tokens"""
        return self._set(penalizeNl=penalizeNl)

    def setNKeep(self, nKeep: int):
        """Set the number of tokens to keep from the initial prompt"""
        return self._set(nKeep=nKeep)

    def setSeed(self, seed: int):
        """Set the RNG seed"""
        return self._set(seed=seed)

    def setNProbs(self, nProbs: int):
        """Set the amount top tokens probabilities to output if greater than 0."""
        return self._set(nProbs=nProbs)

    def setMinKeep(self, minKeep: int):
        """Set the amount of tokens the samplers should return at least (0 = disabled)"""
        return self._set(minKeep=minKeep)

    def setGrammar(self, grammar: str):
        """Set BNF-like grammar to constrain generations"""
        return self._set(grammar=grammar)

    def setPenaltyPrompt(self, penaltyPrompt: str):
        """Override which part of the prompt is penalized for repetition."""
        return self._set(penaltyPrompt=penaltyPrompt)

    def setIgnoreEos(self, ignoreEos: bool):
        """Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)"""
        return self._set(ignoreEos=ignoreEos)

    def setDisableTokenIds(self, disableTokenIds: List[int]):
        """Set the token ids to disable in the completion"""
        return self._set(disableTokenIds=disableTokenIds)

    def setStopStrings(self, stopStrings: List[str]):
        """Set strings upon seeing which token generation is stopped"""
        return self._set(stopStrings=stopStrings)

    def setSamplers(self, samplers: List[str]):
        """Set which samplers to use for token generation in the given order"""
        return self._set(samplers=samplers)

    def setUseChatTemplate(self, useChatTemplate: bool):
        """Set whether generate should apply a chat template"""
        return self._set(useChatTemplate=useChatTemplate)

    def setNParallel(self, nParallel: int):
        """Sets the number of parallel processes for decoding. This is an alias for `setBatchSize`."""
        return self.setBatchSize(nParallel)

    def setLogVerbosity(self, logVerbosity: int):
        """Set the log verbosity level"""
        return self._set(logVerbosity=logVerbosity)

    def setDisableLog(self, disableLog: bool):
        """Whether to disable logging"""
        return self._set(disableLog=disableLog)

    # -------- JAVA SETTERS --------
    def setTokenIdBias(self, tokenIdBias: Dict[int, float]):
        """Set token id bias"""
        return self._call_java("setTokenIdBias", tokenIdBias)

    def setTokenBias(self, tokenBias: Dict[str, float]):
        """Set token id bias"""
        return self._call_java("setTokenBias", tokenBias)

    # def setLoraAdapters(self, loraAdapters: Dict[str, float]):
    #     """Set LoRA adapters with their scaling factors"""
    #     return self._call_java("setLoraAdapters", loraAdapters)

    def getMetadata(self):
        """Gets the metadata of the model"""
        return self._call_java("getMetadata")
