#  Copyright 2017-2023 John Snow Labs
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
"""Contains classes for the AutoGGUFModel."""
from typing import List, Dict

from sparknlp.common import *


class AutoGGUFModel(AnnotatorModel, HasBatchedAnnotate):
    """
    Annotator that uses the llama.cpp library to generate text completions with large language
    models.

    For settable parameters, and their explanations, see the parameters of this class and refer to
    the llama.cpp documentation of
    `server.cpp <https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server>`__
    for more information.

    If the parameters are not set, the annotator will default to use the parameters provided by
    the model.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> auto_gguf_model = AutoGGUFModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("completions")

    The default model is ``"phi3.5_mini_4k_instruct_q4_gguf"``, if no name is provided.

    For extended examples of usage, see the
    `AutoGGUFModelTest <https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFModelTest.scala>`__
    and the
    `example notebook <https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFModel.ipynb>`__.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    nThreads
        Set the number of threads to use during generation
    nThreadsDraft
        Set the number of threads to use during draft generation
    nThreadsBatch
        Set the number of threads to use during batch and prompt processing
    nThreadsBatchDraft
        Set the number of threads to use during batch and prompt processing
    nCtx
        Set the size of the prompt context
    nBatch
        Set the logical batch size for prompt processing (must be >=32 to use BLAS)
    nUbatch
        Set the physical batch size for prompt processing (must be >=32 to use BLAS)
    nDraft
        Set the number of tokens to draft for speculative decoding
    nChunks
        Set the maximal number of chunks to process
    nSequences
        Set the number of sequences to decode
    pSplit
        Set the speculative decoding split probability
    nGpuLayers
        Set the number of layers to store in VRAM (-1 - use default)
    nGpuLayersDraft
        Set the number of layers to store in VRAM for the draft model (-1 - use default)
    gpuSplitMode
        Set how to split the model across GPUs
    mainGpu
        Set the main GPU that is used for scratch and small tensors.
    tensorSplit
        Set how split tensors should be distributed across GPUs
    grpAttnN
        Set the group-attention factor
    grpAttnW
        Set the group-attention width
    ropeFreqBase
        Set the RoPE base frequency, used by NTK-aware scaling
    ropeFreqScale
        Set the RoPE frequency scaling factor, expands context by a factor of 1/N
    yarnExtFactor
        Set the YaRN extrapolation mix factor
    yarnAttnFactor
        Set the YaRN scale sqrt(t) or attention magnitude
    yarnBetaFast
        Set the YaRN low correction dim or beta
    yarnBetaSlow
        Set the YaRN high correction dim or alpha
    yarnOrigCtx
        Set the YaRN original context size of model
    defragmentationThreshold
        Set the KV cache defragmentation threshold
    numaStrategy
        Set optimization strategies that help on some NUMA systems (if available)
    ropeScalingType
        Set the RoPE frequency scaling method, defaults to linear unless specified by the model
    poolingType
        Set the pooling type for embeddings, use model default if unspecified
    modelDraft
        Set the draft model for speculative decoding
    modelAlias
        Set a model alias
    lookupCacheStaticFilePath
        Set path to static lookup cache to use for lookup decoding (not updated by generation)
    lookupCacheDynamicFilePath
        Set path to dynamic lookup cache to use for lookup decoding (updated by generation)
    embedding
        Whether to load model with embedding support
    flashAttention
        Whether to enable Flash Attention
    inputPrefixBos
        Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string
    useMmap
        Whether to use memory-map model (faster load but may increase pageouts if not using mlock)
    useMlock
        Whether to force the system to keep model in RAM rather than swapping or compressing
    noKvOffload
        Whether to disable KV offload
    systemPrompt
        Set a system prompt to use
    chatTemplate
        The chat template to use
    inputPrefix
        Set the prompt to start generation with
    inputSuffix
        Set a suffix for infilling
    cachePrompt
        Whether to remember the prompt to avoid reprocessing it
    nPredict
        Set the number of tokens to predict
    topK
        Set top-k sampling
    topP
        Set top-p sampling
    minP
        Set min-p sampling
    tfsZ
        Set tail free sampling, parameter z
    typicalP
        Set locally typical sampling, parameter p
    temperature
        Set the temperature
    dynatempRange
        Set the dynamic temperature range
    dynatempExponent
        Set the dynamic temperature exponent
    repeatLastN
        Set the last n tokens to consider for penalties
    repeatPenalty
        Set the penalty of repeated sequences of tokens
    frequencyPenalty
        Set the repetition alpha frequency penalty
    presencePenalty
        Set the repetition alpha presence penalty
    miroStat
        Set MiroStat sampling strategies.
    mirostatTau
        Set the MiroStat target entropy, parameter tau
    mirostatEta
        Set the MiroStat learning rate, parameter eta
    penalizeNl
        Whether to penalize newline tokens
    nKeep
        Set the number of tokens to keep from the initial prompt
    seed
        Set the RNG seed
    nProbs
        Set the amount top tokens probabilities to output if greater than 0.
    minKeep
        Set the amount of tokens the samplers should return at least (0 = disabled)
    grammar
        Set BNF-like grammar to constrain generations
    penaltyPrompt
        Override which part of the prompt is penalized for repetition.
    ignoreEos
        Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)
    disableTokenIds
        Set the token ids to disable in the completion
    stopStrings
        Set strings upon seeing which token generation is stopped
    samplers
        Set which samplers to use for token generation in the given order
    useChatTemplate
        Set whether or not generate should apply a chat template


    Notes
    -----
    To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
    the number of GPU layers with the `setNGpuLayers` method.

    When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
    according to your hardware to avoid out-of-memory errors.

    References
    ----------
    - `Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
     <https://arxiv.org/abs/1910.13461>`__
    - https://github.com/pytorch/fairseq

    **Paper Abstract:**
    *We present BART, a denoising autoencoder for pretraining sequence-to-sequence models.
    BART is trained by (1) corrupting text with an arbitrary noising function, and (2)
    learning a model to reconstruct the original text. It uses a standard Tranformer-based
    neural machine translation architecture which, despite its simplicity, can be seen as
    generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder),
    and many other more recent pretraining schemes. We evaluate a number of noising approaches,
    finding the best performance by both randomly shuffling the order of the original sentences
    and using a novel in-filling scheme, where spans of text are replaced with a single mask token.
    BART is particularly effective when fine tuned for text generation but also works well for
    comprehension tasks. It matches the performance of RoBERTa with comparable training resources
    on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue,
    question answering, and summarization tasks, with gains of up to 6 ROUGE. BART also provides
    a 1.1 BLEU increase over a back-translation system for machine translation, with only target
    language pretraining. We also report ablation experiments that replicate other pretraining
    schemes within the BART framework, to better measure which factors most influence end-task performance.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> autoGGUFModel = AutoGGUFModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("completions") \\
    ...     .setBatchSize(4) \\
    ...     .setNPredict(20) \\
    ...     .setNGpuLayers(99) \\
    ...     .setTemperature(0.4) \\
    ...     .setTopK(40) \\
    ...     .setTopP(0.9) \\
    ...     .setPenalizeNl(True)
    >>> pipeline = Pipeline().setStages([document, autoGGUFModel])
    >>> data = spark.createDataFrame([["Hello, I am a"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("completions").show(truncate = False)
    +-----------------------------------------------------------------------------------------------------------------------------------+
    |completions                                                                                                                        |
    +-----------------------------------------------------------------------------------------------------------------------------------+
    |[{document, 0, 78,  new user.  I am currently working on a project and I need to create a list of , {prompt -> Hello, I am a}, []}]|
    +-----------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "AutoGGUFModel"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    # -------- MODEl PARAMETERS --------
    nThreads = Param(Params._dummy(), "nThreads", "Set the number of threads to use during generation",
                     typeConverter=TypeConverters.toInt)
    nThreadsDraft = Param(Params._dummy(), "nThreadsDraft", "Set the number of threads to use during draft generation",
                          typeConverter=TypeConverters.toInt)
    nThreadsBatch = Param(Params._dummy(), "nThreadsBatch",
                          "Set the number of threads to use during batch and prompt processing",
                          typeConverter=TypeConverters.toInt)
    nThreadsBatchDraft = Param(Params._dummy(), "nThreadsBatchDraft",
                               "Set the number of threads to use during batch and prompt processing",
                               typeConverter=TypeConverters.toInt)
    nCtx = Param(Params._dummy(), "nCtx", "Set the size of the prompt context", typeConverter=TypeConverters.toInt)
    nBatch = Param(Params._dummy(), "nBatch",
                   "Set the logical batch size for prompt processing (must be >=32 to use BLAS)",
                   typeConverter=TypeConverters.toInt)
    nUbatch = Param(Params._dummy(), "nUbatch",
                    "Set the physical batch size for prompt processing (must be >=32 to use BLAS)",
                    typeConverter=TypeConverters.toInt)
    nDraft = Param(Params._dummy(), "nDraft", "Set the number of tokens to draft for speculative decoding",
                   typeConverter=TypeConverters.toInt)
    nChunks = Param(Params._dummy(), "nChunks", "Set the maximal number of chunks to process",
                    typeConverter=TypeConverters.toInt)
    nSequences = Param(Params._dummy(), "nSequences", "Set the number of sequences to decode",
                       typeConverter=TypeConverters.toInt)
    pSplit = Param(Params._dummy(), "pSplit", "Set the speculative decoding split probability",
                   typeConverter=TypeConverters.toFloat)
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
    tensorSplit = Param(Params._dummy(), "tensorSplit", "Set how split tensors should be distributed across GPUs",
                        typeConverter=TypeConverters.toListFloat)
    grpAttnN = Param(Params._dummy(), "grpAttnN", "Set the group-attention factor", typeConverter=TypeConverters.toInt)
    grpAttnW = Param(Params._dummy(), "grpAttnW", "Set the group-attention width", typeConverter=TypeConverters.toInt)
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
    #   - UNSPECIFIED: Don't use any scaling
    #   - LINEAR: Linear scaling
    #   - YARN: YaRN RoPE scaling
    ropeScalingType = Param(Params._dummy(), "ropeScalingType",
                            "Set the RoPE frequency scaling method, defaults to linear unless specified by the model",
                            typeConverter=TypeConverters.toString)
    # Set the pooling type for embeddings, use model default if unspecified
    #
    #   - 0 UNSPECIFIED: Don't use any pooling
    #   - 1 MEAN: Mean Pooling
    #   - 2 CLS: CLS Pooling
    poolingType = Param(Params._dummy(), "poolingType",
                        "Set the pooling type for embeddings, use model default if unspecified",
                        typeConverter=TypeConverters.toString)
    modelDraft = Param(Params._dummy(), "modelDraft", "Set the draft model for speculative decoding",
                       typeConverter=TypeConverters.toString)
    modelAlias = Param(Params._dummy(), "modelAlias", "Set a model alias", typeConverter=TypeConverters.toString)
    lookupCacheStaticFilePath = Param(Params._dummy(), "lookupCacheStaticFilePath",
                                      "Set path to static lookup cache to use for lookup decoding (not updated by generation)",
                                      typeConverter=TypeConverters.toString)
    lookupCacheDynamicFilePath = Param(Params._dummy(), "lookupCacheDynamicFilePath",
                                       "Set path to dynamic lookup cache to use for lookup decoding (updated by generation)",
                                       typeConverter=TypeConverters.toString)
    # loraAdapters = new StructFeature[Map[String, Float]](this, "loraAdapters")
    embedding = Param(Params._dummy(), "embedding", "Whether to load model with embedding support",
                      typeConverter=TypeConverters.toBoolean)
    flashAttention = Param(Params._dummy(), "flashAttention", "Whether to enable Flash Attention",
                           typeConverter=TypeConverters.toBoolean)
    inputPrefixBos = Param(Params._dummy(), "inputPrefixBos",
                           "Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string",
                           typeConverter=TypeConverters.toBoolean)
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

    def setNThreadsDraft(self, nThreadsDraft: int):
        """Set the number of threads to use during draft generation"""
        return self._set(nThreadsDraft=nThreadsDraft)

    def setNThreadsBatch(self, nThreadsBatch: int):
        """Set the number of threads to use during batch and prompt processing"""
        return self._set(nThreadsBatch=nThreadsBatch)

    def setNThreadsBatchDraft(self, nThreadsBatchDraft: int):
        """Set the number of threads to use during batch and prompt processing"""
        return self._set(nThreadsBatchDraft=nThreadsBatchDraft)

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

    def setNChunks(self, nChunks: int):
        """Set the maximal number of chunks to process"""
        return self._set(nChunks=nChunks)

    def setNSequences(self, nSequences: int):
        """Set the number of sequences to decode"""
        return self._set(nSequences=nSequences)

    def setPSplit(self, pSplit: float):
        """Set the speculative decoding split probability"""
        return self._set(pSplit=pSplit)

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

    def setTensorSplit(self, tensorSplit: List[float]):
        """Set how split tensors should be distributed across GPUs"""
        return self._set(tensorSplit=tensorSplit)

    def setGrpAttnN(self, grpAttnN: int):
        """Set the group-attention factor"""
        return self._set(grpAttnN=grpAttnN)

    def setGrpAttnW(self, grpAttnW: int):
        """Set the group-attention width"""
        return self._set(grpAttnW=grpAttnW)

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
        """Set optimization strategies that help on some NUMA systems (if available)"""
        return self._set(numaStrategy=numaStrategy)

    def setRopeScalingType(self, ropeScalingType: str):
        """Set the RoPE frequency scaling method, defaults to linear unless specified by the model"""
        return self._set(ropeScalingType=ropeScalingType)

    def setPoolingType(self, poolingType: bool):
        """Set the pooling type for embeddings, use model default if unspecified"""
        return self._set(poolingType=poolingType)

    def setModelDraft(self, modelDraft: str):
        """Set the draft model for speculative decoding"""
        return self._set(modelDraft=modelDraft)

    def setModelAlias(self, modelAlias: str):
        """Set a model alias"""
        return self._set(modelAlias=modelAlias)

    def setLookupCacheStaticFilePath(self, lookupCacheStaticFilePath: str):
        """Set path to static lookup cache to use for lookup decoding (not updated by generation)"""
        return self._set(lookupCacheStaticFilePath=lookupCacheStaticFilePath)

    def setLookupCacheDynamicFilePath(self, lookupCacheDynamicFilePath: str):
        """Set path to dynamic lookup cache to use for lookup decoding (updated by generation)"""
        return self._set(lookupCacheDynamicFilePath=lookupCacheDynamicFilePath)

    def setEmbedding(self, embedding: bool):
        """Whether to load model with embedding support"""
        return self._set(embedding=embedding)

    def setFlashAttention(self, flashAttention: bool):
        """Whether to enable Flash Attention"""
        return self._set(flashAttention=flashAttention)

    def setInputPrefixBos(self, inputPrefixBos: bool):
        """Whether to add prefix BOS to user inputs, preceding the `--in-prefix` bool"""
        return self._set(inputPrefixBos=inputPrefixBos)

    def setUseMmap(self, useMmap: bool):
        """Whether to use memory-map model (faster load but may increase pageouts if not using mlock)"""
        return self._set(useMmap=useMmap)

    def setUseMlock(self, useMlock: bool):
        """Whether to force the system to keep model in RAM rather than swapping or compressing"""
        return self._set(useMlock=useMlock)

    def setNoKvOffload(self, noKvOffload: bool):
        """Whether to disable KV offload"""
        return self._set(noKvOffload=noKvOffload)

    def setSystemPrompt(self, systemPrompt: bool):
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

    def setGrammar(self, grammar: bool):
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

    # -------- JAVA SETTERS --------
    def setTokenIdBias(self, tokenIdBias: Dict[int, float]):
        """Set token id bias"""
        return self._call_java("setTokenIdBias", tokenIdBias)

    def setTokenBias(self, tokenBias: Dict[str, float]):
        """Set token id bias"""
        return self._call_java("setTokenBias", tokenBias)

    def setLoraAdapters(self, loraAdapters: Dict[str, float]):
        """Set token id bias"""
        return self._call_java("setLoraAdapters", loraAdapters)

    def getMetadata(self):
        """Gets the metadata of the model"""
        return self._call_java("getMetadata")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel", java_model=None):
        super(AutoGGUFModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        # self._setDefault()

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
        AutoGGUFModel
            The restored model
        """
        from sparknlp.internal import _AutoGGUFLoader
        jModel = _AutoGGUFLoader(folder, spark_session._jsparkSession)._java_obj
        return AutoGGUFModel(java_model=jModel)

    @staticmethod
    def pretrained(name="phi3.5_mini_4k_instruct_q4_gguf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "phi3.5_mini_4k_instruct_q4_gguf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        AutoGGUFModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AutoGGUFModel, name, lang, remote_loc)
