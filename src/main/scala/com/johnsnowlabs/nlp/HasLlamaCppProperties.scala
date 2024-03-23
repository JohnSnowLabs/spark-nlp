package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{
  BooleanParam,
  DoubleArrayParam,
  FloatParam,
  IntArrayParam,
  IntParam,
  Param,
  StringArrayParam
}

/** Parameters to configure beam search text generation. */
trait HasLlamaCppProperties {
  this: ParamsAndFeaturesWritable =>

  // ---------------- MODEL PARAMETERS ----------------
  /** @group param */
  val nThreads = new IntParam(
    this,
    "nThreads",
    "Set the number of threads to use during generation (default: 8)")

  /** @group param */
  val nThreadsDraft = new IntParam(
    this,
    "nThreadsDraft",
    "Set the number of threads to use during draft generation (default: same as nThreads)")

  /** @group param */
  val nThreadsBatch = new IntParam(
    this,
    "nThreadsBatch",
    "Set the number of threads to use during batch and prompt processing (default: same as {@link #setNThreads(int)})")

  /** @group param */
  val nThreadsBatchDraft = new IntParam(
    this,
    "nThreadsBatchDraft",
    "Set the number of threads to use during batch and prompt processing (default: same as {@link #setNThreadsDraft(int)})")

  /** @group param */
  val nCtx = new IntParam(
    this,
    "nCtx",
    "Set the size of the prompt context (default: 512, 0 = loaded from model)")

  /** @group param */
  val nBatch = new IntParam(
    this,
    "nBatch",
    "Set the logical batch size for prompt processing (must be >=32 to use BLAS)")

  /** @group param */
  val nUbatch = new IntParam(
    this,
    "nUbatch",
    "Set the physical batch size for prompt processing (must be >=32 to use BLAS)")

  /** @group param */
  val nDraft = new IntParam(
    this,
    "nDraft",
    "Set the number of tokens to draft for speculative decoding (default: 5)")

  /** @group param */
  val nChunks = new IntParam(
    this,
    "nChunks",
    "Set the maximal number of chunks to process (default: -1, -1 = all)")

  /** @group param */
  val nParallel =
    new IntParam(this, "nParallel", "Set the number of parallel sequences to decode (default: 1)")

  /** @group param */
  val nSequences =
    new IntParam(this, "nSequences", "Set the number of sequences to decode (default: 1)")

  /** @group param */
  val pSplit = new FloatParam(
    this,
    "pSplit",
    "Set the speculative decoding split probability (default: 0.1)")

  /** @group param */
  val nGpuLayers = new IntParam(
    this,
    "nGpuLayers",
    "Set the number of layers to store in VRAM (-1 - use default)")

  /** @group param */
  val nGpuLayersDraft = new IntParam(
    this,
    "nGpuLayersDraft",
    "Set the number of layers to store in VRAM for the draft model (-1 - use default)")

  /** Set how to split the model across GPUs
    *
    * 0. NONE: No GPU split
    *   1. LAYER: Split the model across GPUs by layer 2. ROW: Split the model across GPUs by rows
    *
    * @group param
    */
  /** @group param */
  val gpuSplitMode = new IntParam(this, "gpuSplitMode", "Set how to split the model across GPUs")

  /** @group param */
  val mainGpu = new IntParam(
    this,
    "mainGpu",
    "Set the GPU that is used for scratch and small tensors"
  ) // TODO: what does that even mean?
  /** @group param */
  val tensorSplit = new DoubleArrayParam(
    this,
    "tensorSplit",
    "Set how split tensors should be distributed across GPUs")

  /** @group param */
  val nBeams =
    new IntParam(this, "nBeams", "Set usage of beam search of given width if non-zero.")

  /** @group param */
  val grpAttnN = new IntParam(this, "grpAttnN", "Set the group-attention factor (default: 1)")

  /** @group param */
  val grpAttnW = new IntParam(this, "grpAttnW", "Set the group-attention width (default: 512.0)")

  /** @group param */
  val ropeFreqBase = new FloatParam(
    this,
    "ropeFreqBase",
    "Set the RoPE base frequency, used by NTK-aware scaling (default: loaded from model)")

  /** @group param */
  val ropeFreqScale = new FloatParam(
    this,
    "ropeFreqScale",
    "Set the RoPE frequency scaling factor, expands context by a factor of 1/N")

  /** @group param */
  val yarnExtFactor = new FloatParam(
    this,
    "yarnExtFactor",
    "Set the YaRN extrapolation mix factor (default: 1.0, 0.0 = full interpolation)")

  /** @group param */
  val yarnAttnFactor = new FloatParam(
    this,
    "yarnAttnFactor",
    "Set the YaRN scale sqrt(t) or attention magnitude (default: 1.0)")

  /** @group param */
  val yarnBetaFast = new FloatParam(
    this,
    "yarnBetaFast",
    "Set the YaRN low correction dim or beta (default: 32.0)")

  /** @group param */
  val yarnBetaSlow = new FloatParam(
    this,
    "yarnBetaSlow",
    "Set the YaRN high correction dim or alpha (default: 1.0)")

  /** @group param */
  val yarnOrigCtx = new IntParam(
    this,
    "yarnOrigCtx",
    "Set the YaRN original context size of model (default: 0 = model training context size)")

  /** @group param */
  val defragThold = new FloatParam(
    this,
    "defragThold",
    "Set the KV cache defragmentation threshold (default: -1.0, &lt; 0 - disabled)")

  /** Set optimization strategies that help on some NUMA systems (if available)
    *
    * Available Strategies:
    *
    *   - 0 DISABLED: No NUMA optimizations
    *   - 1 DISTRIBUTE: spread execution evenly over all
    *   - 2 ISOLATE: only spawn threads on CPUs on the node that execution started on
    *   - 3 NUMA_CTL: use the CPU map provided by numactl
    *   - 4 MIRROR: TODO
    *
    * @group param
    */
  val numaStrategy = new IntParam(
    this,
    "numaStrategy",
    "Set optimization strategies that help on some NUMA systems (if available)")

  /** Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    *
    *   - 0 UNSPECIFIED: TODO
    *   - 1 LINEAR: Linear scaling
    *   - 2 YARN: TODO
    * @group param
    */
  val ropeScalingType = new IntParam(
    this,
    "ropeScalingType",
    "Set the RoPE frequency scaling method, defaults to linear unless specified by the model")

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - 0 UNSPECIFIED: TODO
    *   - 1 MEAN: Mean Pooling
    *   - 2 CLS: CLS Pooling
    *
    * @group param
    */
  val poolingType = new IntParam(
    this,
    "poolingType",
    "Set the pooling type for embeddings, use model default if unspecified")
  //  model = new Param[String](this, "model", "Set the model file path to load (default: models/7B/ggml-model-f16.gguf)")
  /** @group param */
  val modelDraft = new Param[String](
    this,
    "modelDraft",
    "Set the draft model for speculative decoding (default: unused)")
  //  modelAlias = new Param[String](this, "modelAlias", "Set a model alias")
  /** @group param
    * TODO: Needed?
    */
  val lookupCacheStatic = new Param[String](
    this,
    "lookupCacheStatic",
    "Set path to static lookup cache to use for lookup decoding (not updated by generation)")

  /** @group param
    * TODO: Needed?
    */
  val lookupCacheDynamic = new Param[String](
    this,
    "lookupCacheDynamic",
    "Set path to dynamic lookup cache to use for lookup decoding (updated by generation)")
  //    * Set LoRA adapters to use (implies --no-mmap).
  /** @group param */
  //    val loraAdapters = new Map<String, Param(this, "Float", "The key is expected to be a file path, the values are expected to be scales.")
  /** @group param */
  val loraBase = new Param[String](
    this,
    "loraBase",
    "Set an optional model to use as a base for the layers modified by the LoRA adapter")

  /** @group param */
  val embedding =
    new BooleanParam(this, "embedding", "Whether to load model with embedding support")

  /** @group param */
  val contBatching = new BooleanParam(
    this,
    "contBatching",
    "Whether to enable continuous batching (also called dynamic batching) (default: disabled)")

  /** @group param */
  val flashAttention = new BooleanParam(
    this,
    "flashAttention",
    "Whether to enable Flash Attention (default: disabled)")

  /** @group param */
  val inputPrefixBos = new BooleanParam(
    this,
    "inputPrefixBos",
    "Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string")

  /** @group param */
  val useMmap = new BooleanParam(
    this,
    "useMmap",
    "Whether to use memory-map model (faster load but may increase pageouts if not using mlock)")

  /** @group param */
  val useMlock = new BooleanParam(
    this,
    "useMlock",
    "Whether to force the system to keep model in RAM rather than swapping or compressing")

  /** @group param */
  val noKvOffload = new BooleanParam(this, "noKvOffload", "Whether to disable KV offload")

  /** @group param */
  val systemPrompt = new Param[String](this, "systemPrompt", "Set a system prompt to use")

  /** @group param */
  val chatTemplate =
    new Param[String](this, "chatTemplate", "The chat template to use (default: empty)")

  /** Set the number of threads to use during generation (default: 8)
    *
    * @group setParam
    */
  def setNThreads(nThreads: Int): this.type = { set(this.nThreads, nThreads) }

  /** Set the number of threads to use during draft generation (default: same as nThreads)
    *
    * @group setParam
    */
  def setNThreadsDraft(nThreadsDraft: Int): this.type = { set(this.nThreadsDraft, nThreadsDraft) }

  /** Set the number of threads to use during batch and prompt processing (default: same as
    * nThreads)
    *
    * @group setParam
    */
  def setNThreadsBatch(nThreadsBatch: Int): this.type = { set(this.nThreadsBatch, nThreadsBatch) }

  /** Set the number of threads to use during batch and prompt processing (default: same as
    * nThreads)
    *
    * @group setParam
    */
  def setNThreadsBatchDraft(nThreadsBatchDraft: Int): this.type = {
    set(this.nThreadsBatchDraft, nThreadsBatchDraft)
  }

  /** Set the size of the prompt context (default: 512, 0 = loaded from model)
    *
    * @group setParam
    */
  def setNCtx(nCtx: Int): this.type = { set(this.nCtx, nCtx) }

  /** Set the logical batch size for prompt processing (must be >=32 to use BLAS)
    *
    * @group setParam
    */
  def setNBatch(nBatch: Int): this.type = { set(this.nBatch, nBatch) }

  /** Set the physical batch size for prompt processing (must be >=32 to use BLAS)
    *
    * @group setParam
    */
  def setNUbatch(nUbatch: Int): this.type = { set(this.nUbatch, nUbatch) }

  /** Set the number of tokens to draft for speculative decoding (default: 5)
    *
    * @group setParam
    */
  def setNDraft(nDraft: Int): this.type = { set(this.nDraft, nDraft) }

  /** Set the maximal number of chunks to process (default: -1, -1 = all)
    *
    * @group setParam
    */
  def setNChunks(nChunks: Int): this.type = { set(this.nChunks, nChunks) }

  /** Set the number of parallel sequences to decode (default: 1)
    *
    * @group setParam
    */
  def setNParallel(nParallel: Int): this.type = { set(this.nParallel, nParallel) }

  /** Set the number of sequences to decode (default: 1)
    *
    * @group setParam
    */
  def setNSequences(nSequences: Int): this.type = { set(this.nSequences, nSequences) }

  /** Set the speculative decoding split probability (default: 0.1)
    *
    * @group setParam
    */
  def setPSplit(pSplit: Float): this.type = { set(this.pSplit, pSplit) }

  /** Set the number of layers to store in VRAM (-1 - use default)
    *
    * @group setParam
    */
  def setNGpuLayers(nGpuLayers: Int): this.type = { set(this.nGpuLayers, nGpuLayers) }

  /** Set the number of layers to store in VRAM for the draft model (-1 - use default)
    *
    * @group setParam
    */
  def setNGpuLayersDraft(nGpuLayersDraft: Int): this.type = { set(this.nGpuLayersDraft, nGpuLayersDraft) }

  /** Set how to split the model across GPUs
    *
    * 0. NONE: No GPU split
    *   1. LAYER: Split the model across GPUs by layer 2. ROW: Split the model across GPUs by rows
    *
    * @group setParam
    */
  def setSplitMode(splitMode: Int): this.type = { set(this.gpuSplitMode, splitMode) }

  /** Set the GPU that is used for scratch and small tensors
    *
    * @group setParam
    */
  def setMainGpu(mainGpu: Int): this.type = { set(this.mainGpu, mainGpu) }

  /** Set how split tensors should be distributed across GPUs
    *
    * @group setParam
    */
  def setTensorSplit(tensorSplit: Array[Double]): this.type = { set(this.tensorSplit, tensorSplit) }

  /** Set usage of beam search of given width if non-zero.
    *
    * @group setParam
    */
  def setNBeams(nBeams: Int): this.type = { set(this.nBeams, nBeams) }

  /** Set the group-attention factor (default: 1)
    *
    * @group setParam
    */
  def setGrpAttnN(grpAttnN: Int): this.type = { set(this.grpAttnN, grpAttnN) }

  /** Set the group-attention width (default: 512.0)
    *
    * @group setParam
    */
  def setGrpAttnW(grpAttnW: Int): this.type = { set(this.grpAttnW, grpAttnW) }

  /** Set the RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
    *
    * @group setParam
    */
  def setRopeFreqBase(ropeFreqBase: Float): this.type = { set(this.ropeFreqBase, ropeFreqBase) }

  /** Set the RoPE frequency scaling factor, expands context by a factor of 1/N
    *
    * @group setParam
    */
  def setRopeFreqScale(ropeFreqScale: Float): this.type = { set(this.ropeFreqScale, ropeFreqScale) }

  /** Set the YaRN extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
    *
    * @group setParam
    */
  def setYarnExtFactor(yarnExtFactor: Float): this.type = { set(this.yarnExtFactor, yarnExtFactor) }

  /** Set the YaRN scale sqrt(t) or attention magnitude (default: 1.0)
    *
    * @group setParam
    */
  def setYarnAttnFactor(yarnAttnFactor: Float): this.type = { set(this.yarnAttnFactor, yarnAttnFactor) }

  /** Set the YaRN low correction dim or beta (default: 32.0)
    *
    * @group setParam
    */
  def setYarnBetaFast(yarnBetaFast: Float): this.type = { set(this.yarnBetaFast, yarnBetaFast) }

  /** Set the YaRN high correction dim or alpha (default: 1.0)
    *
    * @group setParam
    */
  def setYarnBetaSlow(yarnBetaSlow: Float): this.type = { set(this.yarnBetaSlow, yarnBetaSlow) }

  /** Set the YaRN original context size of model (default: 0 = model training context size)
    *
    * @group setParam
    */
  def setYarnOrigCtx(yarnOrigCtx: Int): this.type = { set(this.yarnOrigCtx, yarnOrigCtx) }

  /** Set the KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
    *
    * @group setParam
    */
  def setDefragmentationThreshold(defragThold: Float): this.type = { set(this.defragThold, defragThold) }

  /** Set optimization strategies that help on some NUMA systems (if available)
    *
    * Available Strategies:
    *
    *   - 0 DISABLED: No NUMA optimizations
    *   - 1 DISTRIBUTE: spread execution evenly over all
    *   - 2 ISOLATE: only spawn threads on CPUs on the node that execution started on
    *   - 3 NUMA_CTL: use the CPU map provided by numactl
    *   - 4 MIRROR: TODO
    *
    * @group setParam
    */
  def setNuma(numa: Int): this.type = { set(this.numaStrategy, numa) }

  /** Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    *
    *   - 0 UNSPECIFIED: TODO
    *   - 1 LINEAR: Linear scaling
    *   - 2 YARN: TODO
    * @group setParam
    */
  def setRopeScalingType(ropeScalingType: Int): this.type = { set(this.ropeScalingType, ropeScalingType) }

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - 0 UNSPECIFIED: TODO
    *   - 1 MEAN: Mean Pooling
    *   - 2 CLS: CLS Pooling
    *
    * @group setParam
    */
  def setPoolingType(poolingType: Int): this.type = { set(this.poolingType, poolingType) }

  /** Set the draft model for speculative decoding (default: unused)
    *
    * @group setParam
    */
  def setModelDraft(modelDraft: String): this.type = { set(this.modelDraft, modelDraft) }

  /** Set a model alias
    *
    * @group setParam
    */
  def setLookupCacheStaticFilePath(lookupCacheStatic: String): this.type = {
    set(this.lookupCacheStatic, lookupCacheStatic)
  }

  /** Set a model alias
    *
    * @group setParam
    */
  def setLookupCacheDynamicFilePath(lookupCacheDynamic: String): this.type = {
    set(this.lookupCacheDynamic, lookupCacheDynamic)
  }
  //  def setLoraAdapters(Float: Map<String,> loraAdapters): this.type = { set(this.Float, Float) }
  /** Set an optional model to use as a base for the layers modified by the LoRA adapter
    *
    * @group setParam
    */
  def setLoraBase(loraBase: String): this.type = { set(this.loraBase, loraBase) }

  /** Whether to load model with embedding support
    *
    * @group setParam
    */
  def setEmbedding(embedding: Boolean): this.type = { set(this.embedding, embedding) }

  /** Whether to enable continuous batching (also called dynamic batching) (default: disabled)
    *
    * @group setParam
    */
  def setContinuousBatching(contBatching: Boolean): this.type = { set(this.contBatching, contBatching) }

  /** Whether to enable Flash Attention (default: disabled)
    *
    * @group setParam
    */
  def setFlashAttention(flashAttention: Boolean): this.type = { set(this.flashAttention, flashAttention) }

  /** Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string
    *
    * @group setParam
    */
  def setInputPrefixBos(inputPrefixBos: Boolean): this.type = { set(this.inputPrefixBos, inputPrefixBos) }

  /** Whether to use memory-map model (faster load but may increase pageouts if not using mlock)
    *
    * @group setParam
    */
  def setUseMmap(useMmap: Boolean): this.type = { set(this.useMmap, useMmap) }

  /** Whether to force the system to keep model in RAM rather than swapping or compressing
    *
    * @group setParam
    */
  def setUseMlock(useMlock: Boolean): this.type = { set(this.useMlock, useMlock) }

  /** Whether to disable KV offload
    *
    * @group setParam
    */
  def setNoKvOffload(noKvOffload: Boolean): this.type = { set(this.noKvOffload, noKvOffload) }

  /** Set a system prompt to use
    *
    * @group setParam
    */
  def setSystemPrompt(systemPrompt: String): this.type = { set(this.systemPrompt, systemPrompt) }

  /** The chat template to use (default: empty)
    *
    * @group setParam
    */
  def setChatTemplate(chatTemplate: String): this.type = { set(this.chatTemplate, chatTemplate) }

  // ---------------- GETTERS ----------------
  /** @group getParam */
  def getNThreads: Int = $(nThreads)

  /** @group getParam */
  def getNThreadsDraft: Int = $(nThreadsDraft)

  /** @group getParam */
  def getNThreadsBatch: Int = $(nThreadsBatch)

  /** @group getParam */
  def getNThreadsBatchDraft: Int = $(nThreadsBatchDraft)

  /** @group getParam */
  def getNCtx: Int = $(nCtx)

  /** @group getParam */
  def getNBatch: Int = $(nBatch)

  /** @group getParam */
  def getNUbatch: Int = $(nUbatch)

  /** @group getParam */
  def getNDraft: Int = $(nDraft)

  /** @group getParam */
  def getNChunks: Int = $(nChunks)

  /** @group getParam */
  def getNParallel: Int = $(nParallel)

  /** @group getParam */
  def getNSequences: Int = $(nSequences)

  /** @group getParam */
  def getPSplit: Float = $(pSplit)

  /** @group getParam */
  def getNGpuLayers: Int = $(nGpuLayers)

  /** @group getParam */
  def getNGpuLayersDraft: Int = $(nGpuLayersDraft)

  /** @group getParam */
  def getSplitMode: Int = $(gpuSplitMode)

  /** @group getParam */
  def getMainGpu: Int = $(mainGpu)

  /** @group getParam */
  def getTensorSplit: Array[Double] = $(tensorSplit)

  /** @group getParam */
  def getNBeams: Int = $(nBeams)

  /** @group getParam */
  def getGrpAttnN: Int = $(grpAttnN)

  /** @group getParam */
  def getGrpAttnW: Int = $(grpAttnW)

  /** @group getParam */
  def getRopeFreqBase: Float = $(ropeFreqBase)

  /** @group getParam */
  def getRopeFreqScale: Float = $(ropeFreqScale)

  /** @group getParam */
  def getYarnExtFactor: Float = $(yarnExtFactor)

  /** @group getParam */
  def getYarnAttnFactor: Float = $(yarnAttnFactor)

  /** @group getParam */
  def getYarnBetaFast: Float = $(yarnBetaFast)

  /** @group getParam */
  def getYarnBetaSlow: Float = $(yarnBetaSlow)

  /** @group getParam */
  def getYarnOrigCtx: Int = $(yarnOrigCtx)

  /** @group getParam */
  def getDefragmentationThreshold: Float = $(defragThold)

  /** @group getParam */
  def getNuma: Int = $(numaStrategy)

  /** @group getParam */
  def getRopeScalingType: Int = $(ropeScalingType)

  /** @group getParam */
  def getPoolingType: Int = $(poolingType)

  /** @group getParam */
  def getModelDraft: String = $(modelDraft)

  /** @group getParam */
  def getLookupCacheStaticFilePath: String = $(lookupCacheStatic)

  /** @group getParam */
  def getLookupCacheDynamicFilePath: String = $(lookupCacheDynamic)

  /** @group getParam */
  //  def getLoraAdapters : Map= ???
  /** @group getParam */
  def getLoraBase: String = $(loraBase)

  /** @group getParam */
  def getEmbedding: Boolean = $(embedding)

  /** @group getParam */
  def getContinuousBatching: Boolean = $(contBatching)

  /** @group getParam */
  def getFlashAttention: Boolean = $(flashAttention)

  /** @group getParam */
  def getInputPrefixBos: Boolean = $(inputPrefixBos)

  /** @group getParam */
  def getUseMmap: Boolean = $(useMmap)

  /** @group getParam */
  def getUseMlock: Boolean = $(useMlock)

  /** @group getParam */
  def getNoKvOffload: Boolean = $(noKvOffload)

  /** @group getParam */
  def getSystemPrompt: String = $(systemPrompt)

  /** @group getParam */
  def getChatTemplate: String = $(chatTemplate)

  // ---------------- INFERENCE PARAMETERS ----------------
//  val prompt = new Param[String]("prompt", "")
  /** @group param */
  val inputPrefix = new Param[String](
    this,
    "inputPrefix",
    "Set the prompt to start generation with (default: empty)")

  /** @group param */
  val inputSuffix =
    new Param[String](this, "inputSuffix", "Set a suffix for infilling (default: empty)")

  /** @group param */
  val cachePrompt = new BooleanParam(
    this,
    "cachePrompt",
    "Whether to remember the prompt to avoid reprocessing it")

  /** @group param */
  val nPredict = new IntParam(
    this,
    "nPredict",
    "Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)")

  /** @group param */
  val topK = new IntParam(this, "topK", "Set top-k sampling (default: 40, 0 = disabled)")

  /** @group param */
  val topP = new FloatParam(this, "topP", "Set top-p sampling (default: 0.9, 1.0 = disabled)")

  /** @group param */
  val minP = new FloatParam(this, "minP", "Set min-p sampling (default: 0.1, 0.0 = disabled)")

  /** @group param */
  val tfsZ = new FloatParam(
    this,
    "tfsZ",
    "Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val typicalP = new FloatParam(
    this,
    "typicalP",
    "Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val temperature = new FloatParam(this, "temperature", "Set the temperature (default: 0.8)")

  /** @group param */
  val dynamicTemperatureRange = new FloatParam(
    this,
    "dynatempRange",
    "Set the dynamic temperature range (default: 0.0, 0.0 = disabled)")

  /** @group param */
  val dynamicTemperatureExponent = new FloatParam(
    this,
    "dynatempExponent",
    "Set the dynamic temperature exponent (default: 1.0)")

  /** @group param */
  val repeatLastN = new IntParam(
    this,
    "repeatLastN",
    "Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)")

  /** @group param */
  val repeatPenalty = new FloatParam(
    this,
    "repeatPenalty",
    "Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val frequencyPenalty = new FloatParam(
    this,
    "frequencyPenalty",
    "Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)")

  /** @group param */
  val presencePenalty = new FloatParam(
    this,
    "presencePenalty",
    "Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)")

  /** @group param */
  val miroStat = new IntParam(this, "miroStat", "Set MiroStat sampling strategies.")

  /** @group param */
  val miroStatTau = new FloatParam(
    this,
    "mirostatTau",
    "Set the MiroStat target entropy, parameter tau (default: 5.0)")

  /** @group param */
  val miroStatEta = new FloatParam(
    this,
    "mirostatEta",
    "Set the MiroStat learning rate, parameter eta (default: 0.1)")

  /** @group param */
  val penalizeNl = new BooleanParam(this, "penalizeNl", "Whether to penalize newline tokens")

  /** @group param */
  val nKeep = new IntParam(
    this,
    "nKeep",
    "Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)")

  /** @group param */
  val seed = new IntParam(this, "seed", "Set the RNG seed (default: -1, use random seed for < 0)")

  /** @group param */
  val nProbs = new IntParam(
    this,
    "nProbs",
    "Set the amount top tokens probabilities to output if greater than 0.")

  /** @group param */
  val minKeep = new IntParam(
    this,
    "minKeep",
    "Set the amount of tokens the samplers should return at least (0 = disabled)")

  /** @group param */
  val grammar =
    new Param[String](this, "grammar", "Set BNF-like grammar to constrain generations")

  /** @group param */
  val penaltyPrompt = new Param[String](
    this,
    "penaltyPrompt",
    "Override which part of the prompt is penalized for repetition.")

  /** @group param */
  val ignoreEos = new BooleanParam(
    this,
    "ignoreEos",
    "Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)")
  // Modify the likelihood of tokens appearing in the completion by their id.
// TODO:  val tokenIdBias: Map[Integer, Float]

  // Modify the likelihood of tokens appearing in the completion by their string.
// TODO:  val tokenBias: Map[String, Float]
  // 	 * Set tokens to disable, this corresponds to {@link #setTokenIdBias(Map)} with a value of
  //	 * {@link Float#NEGATIVE_INFINITY}.
  /** @group param */
  val disableTokenIds =
    new IntArrayParam(this, "disableTokenIds", "Set the token ids to disable in the completion")

  /** @group param */
  val stopStrings = new StringArrayParam(
    this,
    "stopStrings",
    "Set strings upon seeing which token generation is stopped")

  // Set which samplers to use for token generation in the given order
  // val samplers = Sampler... samplers // either TOP_K, TFS_Z, TYPICAL_P, TOP_P, MIN_P, TEMPERATURE
  /** @group param */
//  val stream = new BooleanParam(this, "stream", "Whether to stream the output or not")
  /** @group param */
  val useChatTemplate = new BooleanParam(
    this,
    "useChatTemplate",
    "Set whether or not generate should apply a chat template (default: false)")

  /** Set the prompt to start generation with (default: "")
    *
    * @group setParam
    */
  def setInputPrefix(inputPrefix: String): this.type = { set(this.inputPrefix, inputPrefix) }

  /** Set a suffix for infilling (default: "")
    *
    * @group setParam
    */
  def setInputSuffix(inputSuffix: String): this.type = { set(this.inputSuffix, inputSuffix) }

  /** Whether to remember the prompt to avoid reprocessing it
    *
    * @group setParam
    */
  def setCachePrompt(cachePrompt: Boolean): this.type = { set(this.cachePrompt, cachePrompt) }

  /** Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
    *
    * @group setParam
    */
  def setNPredict(nPredict: Int): this.type = { set(this.nPredict, nPredict) }

  /** Set top-k sampling (default: 40, 0 = disabled)
    *
    * @group setParam
    */
  def setTopK(topK: Int): this.type = { set(this.topK, topK) }

  /** Set top-p sampling (default: 0.9, 1.0 = disabled)
    *
    * @group setParam
    */
  def setTopP(topP: Float): this.type = { set(this.topP, topP) }

  /** Set min-p sampling (default: 0.1, 0.0 = disabled)
    *
    * @group setParam
    */
  def setMinP(minP: Float): this.type = { set(this.minP, minP) }

  /** Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
    * @group setParam
    */
  def setTfsZ(tfsZ: Float): this.type = { set(this.tfsZ, tfsZ) }

  /** Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
    *
    * @group setParam
    */
  def setTypicalP(typicalP: Float): this.type = { set(this.typicalP, typicalP) }

  /** Set the temperature (default: 0.8)
    *
    * @group setParam
    */
  def setTemperature(temperature: Float): this.type = { set(this.temperature, temperature) }

  /** Set the dynamic temperature range (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setDynamicTemperatureRange(dynatempRange: Float): this.type = {
    set(this.dynamicTemperatureRange, dynatempRange)
  }
  def setDynamicTemperatureExponent(dynatempExponent: Float): this.type = {
    set(this.dynamicTemperatureExponent, dynatempExponent)
  }

  /** Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)
    *
    * @group setParam
    */
  def setRepeatLastN(repeatLastN: Int): this.type = { set(this.repeatLastN, repeatLastN) }

  /** Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)
    *
    * @group setParam
    */
  def setRepeatPenalty(repeatPenalty: Float): this.type = { set(this.repeatPenalty, repeatPenalty) }

  /** Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setFrequencyPenalty(frequencyPenalty: Float): this.type = {
    set(this.frequencyPenalty, frequencyPenalty)
  }

  /** Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setPresencePenalty(presencePenalty: Float): this.type = { set(this.presencePenalty, presencePenalty) }

  /** Set MiroStat sampling strategies.
    *
    *   - 0 DISABLED: No MiroStat
    *   - 1 V1: MiroStat V1
    *   - 2 V2: MiroStat V2
    *
    * @group setParam
    */
  def setMiroStat(mirostat: Int): this.type = set(this.miroStat, mirostat)

  /** Set the MiroStat target entropy, parameter tau (default: 5.0)
    *
    * @group setParam
    */
  def setMiroStatTau(mirostatTau: Float): this.type = { set(this.miroStatTau, mirostatTau) }

  /** Set the MiroStat learning rate, parameter eta (default: 0.1)
    *
    * @group setParam
    */
  def setMiroStatEta(mirostatEta: Float): this.type = { set(this.miroStatEta, mirostatEta) }

  /** Set whether to penalize newline tokens
    *
    * @group setParam
    */
  def setPenalizeNl(penalizeNl: Boolean): this.type = { set(this.penalizeNl, penalizeNl) }

  /** Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)
    *
    * @group setParam
    */
  def setNKeep(nKeep: Int): this.type = { set(this.nKeep, nKeep) }

  /** Set the RNG seed (default: -1, use random seed for < 0)
    *
    * @group setParam
    */
  def setSeed(seed: Int): this.type = { set(this.seed, seed) }

  /** Set the amount top tokens probabilities to output if greater than 0.
    *
    * @group setParam
    */
  def setNProbs(nProbs: Int): this.type = { set(this.nProbs, nProbs) }

  /** Set the amount of tokens the samplers should return at least (0 = disabled)
    *
    * @group setParam
    */
  def setMinKeep(minKeep: Int): this.type = { set(this.minKeep, minKeep) }

  /** Set BNF-like grammar to constrain generations
    *
    * @group setParam
    */
  def setGrammar(grammar: String): this.type = { set(this.grammar, grammar) }

  /** Override which part of the prompt is penalized for repetition.
    *
    * @group setParam
    */
  def setPenaltyPrompt(penaltyPrompt: String): this.type = { set(this.penaltyPrompt, penaltyPrompt) }

// TODO?  def setPenaltyPrompt(tokens: Array[Int] ): this.type =  {set(this.penaltyPrompt, tokens)}

  /** Set whether to ignore end of stream token and continue generating (implies --logit-bias
    * 2-inf)
    *
    * @group setParam
    */
  def setIgnoreEos(ignoreEos: Boolean): this.type = { set(this.ignoreEos, ignoreEos) }
// TODO: def setTokenIdBias(Float: Map<Integer, > logitBias): this.type =  {set(this.Float, Float)}
// TODO: def setTokenBias(Float: Map<String, > logitBias): this.type =  {set(this.Float, Float)}

  /** Set the token ids to disable in the completion
    *
    * @group setParam
    */
  def setDisableTokenIds(disableTokenIds: Array[Int]): this.type = {
    set(this.disableTokenIds, disableTokenIds)
  }

  /** Set strings upon seeing which token generation is stopped
    *
    * @group setParam
    */
  def setStopStrings(stopStrings: Array[String]): this.type = { set(this.stopStrings, stopStrings) }
//  def setSamplers(samplers: Sampler... ): this.type =  {set(this.samplers, samplers)}

  /** Whether or not to stream the output
    *
    * @group setParam
    */
  def setUseChatTemplate(useChatTemplate: Boolean): this.type = {
    set(this.useChatTemplate, useChatTemplate)
  }

  // ---------------- GETTERS ----------------
  /** @group getParam */
  def getInputPrefix: String = $(inputPrefix)

  /** @group getParam */
  def getInputSuffix: String = $(inputSuffix)

  /** @group getParam */
  def getCachePrompt: Boolean = $(cachePrompt)

  /** @group getParam */
  def getNPredict: Int = $(nPredict)

  /** @group getParam */
  def getTopK: Int = $(topK)

  /** @group getParam */
  def getTopP: Float = $(topP)

  /** @group getParam */
  def getMinP: Float = $(minP)

  /** @group getParam */
  def getTfsZ: Float = $(tfsZ)

  /** @group getParam */
  def getTypicalP: Float = $(typicalP)

  /** @group getParam */
  def getTemperature: Float = $(temperature)

  /** @group getParam */
  def getDynamicTemperatureRange: Float = $(dynamicTemperatureRange)

  /** @group getParam */
  def getDynamicTemperatureExponent: Float = $(dynamicTemperatureExponent)

  /** @group getParam */
  def getRepeatLastN: Int = $(repeatLastN)

  /** @group getParam */
  def getRepeatPenalty: Float = $(repeatPenalty)

  /** @group getParam */
  def getFrequencyPenalty: Float = $(frequencyPenalty)

  /** @group getParam */
  def getPresencePenalty: Float = $(presencePenalty)

  /** @group getParam */
  def getMiroStat: Int = $(miroStat)

  /** @group getParam */
  def getMiroStatTau: Float = $(miroStatTau)

  /** @group getParam */
  def getMiroStatEta: Float = $(miroStatEta)

  /** @group getParam */
  def getPenalizeNl: Boolean = $(penalizeNl)

  /** @group getParam */
  def getNKeep: Int = $(nKeep)

  /** @group getParam */
  def getSeed: Int = $(seed)

  /** @group getParam */
  def getNProbs: Int = $(nProbs)

  /** @group getParam */
  def getMinKeep: Int = $(minKeep)

  /** @group getParam */
  def getGrammar: String = $(grammar)

  /** @group getParam */
  def getPenaltyPrompt: String = $(penaltyPrompt)

  /** @group getParam */
  def getIgnoreEos: Boolean = $(ignoreEos)

  /** @group getParam */
  //  def getTokenIdBias = ???
  /** @group getParam */
  //  def getTokenBias =  ???
  /** @group getParam */
  def getDisableTokenIds: Array[Int] = $(disableTokenIds)

  /** @group getParam */
  def getStopStrings: Array[String] = $(stopStrings)

  /** @group getParam */
  //  def getSamplers : Sampler=  (samplers
  /** @group getParam */
  def getUseChatTemplate: Boolean = $(useChatTemplate)

  setDefault(
    inputPrefix -> "",
    inputSuffix -> "",
    cachePrompt -> false,
    nPredict -> -1,
    topK -> 40,
    topP -> 0.9f,
    minP -> 0.1f,
    tfsZ -> 1.0f,
    typicalP -> 1.0f,
    temperature -> 0.8f,
    dynamicTemperatureRange -> 0.0f,
    dynamicTemperatureExponent -> 1.0f,
    repeatLastN -> 64,
    repeatPenalty -> 1.0f,
    frequencyPenalty -> 0.0f,
    presencePenalty -> 0.0f,
    miroStat -> 0,
    miroStatTau -> 5.0f,
    miroStatEta -> 0.1f,
    penalizeNl -> false,
    nKeep -> 0,
    seed -> -1,
    nProbs -> 0,
    minKeep -> 0,
    grammar -> "",
    penaltyPrompt -> "",
    ignoreEos -> false,
    disableTokenIds -> Array[Int](),
    stopStrings -> Array[String](),
    useChatTemplate -> false,
    nThreads -> 8,
    nThreadsDraft -> 8,
    nThreadsBatch -> 8,
    nThreadsBatchDraft -> 8,
    nCtx -> 512,
    nBatch -> 32,
    nUbatch -> 32,
    nDraft -> 5,
    nChunks -> -1,
    nParallel -> 1,
    nSequences -> 1,
    pSplit -> 0.1f,
    nGpuLayers -> -1,
    nGpuLayersDraft -> -1,
    gpuSplitMode -> 0,
    mainGpu -> 0,
    tensorSplit -> Array[Double](),
    nBeams -> 0,
    grpAttnN -> 1,
    grpAttnW -> 512,
    ropeFreqBase -> 1.0f,
    ropeFreqScale -> 1.0f,
    yarnExtFactor -> 1.0f,
    yarnAttnFactor -> 1.0f,
    yarnBetaFast -> 32.0f,
    yarnBetaSlow -> 1.0f,
    yarnOrigCtx -> 0,
    defragThold -> -1.0f,
    numaStrategy -> 0,
    ropeScalingType -> 0,
    poolingType -> 0,
    modelDraft -> "",
    lookupCacheStatic -> "",
    lookupCacheDynamic -> "",
    loraBase -> "",
    embedding -> false,
    contBatching -> false,
    flashAttention -> false,
    inputPrefixBos -> false,
    useMmap -> false,
    useMlock -> false,
    noKvOffload -> false,
    systemPrompt -> "",
    chatTemplate -> "")
}
