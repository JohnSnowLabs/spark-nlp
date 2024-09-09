package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
import com.johnsnowlabs.nlp.llama.args._
import com.johnsnowlabs.nlp.llama.{InferenceParameters, ModelParameters}
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.jdk.CollectionConverters._

/** Contains settable parameters for the [[AutoGGUFModel]].
  *
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio setParam  1
  * @groupprio getParam  2
  * @groupprio param  3
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
trait HasLlamaCppProperties {
  this: ParamsAndFeaturesWritable with HasProtectedParams =>
  val logger = LoggerFactory.getLogger(this.getClass)
  // ---------------- MODEL PARAMETERS ----------------
  /** @group param */
  val nThreads =
    new IntParam(this, "nThreads", "Set the number of threads to use during generation")

  /** @group param */
  val nThreadsDraft = new IntParam(
    this,
    "nThreadsDraft",
    "Set the number of threads to use during draft generation")

  /** @group param */
  val nThreadsBatch = new IntParam(
    this,
    "nThreadsBatch",
    "Set the number of threads to use during batch and prompt processing")

  /** @group param */
  val nThreadsBatchDraft = new IntParam(
    this,
    "nThreadsBatchDraft",
    "Set the number of threads to use during batch and prompt processing")

  /** @group param */
  val nCtx = new IntParam(this, "nCtx", "Set the size of the prompt context")

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
  val nDraft =
    new IntParam(this, "nDraft", "Set the number of tokens to draft for speculative decoding")

  /** @group param */
  val nChunks = new IntParam(this, "nChunks", "Set the maximal number of chunks to process")

  /** @group param */
  val nSequences =
    new IntParam(this, "nSequences", "Set the number of sequences to decode")

  /** @group param */
  val pSplit = new FloatParam(this, "pSplit", "Set the speculative decoding split probability")

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
    *   - NONE: No GPU split
    *   - LAYER: Split the model across GPUs by layer
    *   - ROW: Split the model across GPUs by rows
    *
    * @group param
    */
  val gpuSplitMode =
    new Param[String](this, "gpuSplitMode", "Set how to split the model across GPUs")

  /** @group param */
  val mainGpu =
    new IntParam(this, "mainGpu", "Set the main GPU that is used for scratch and small tensors.")

  /** @group param */
  val tensorSplit = new DoubleArrayParam(
    this,
    "tensorSplit",
    "Set how split tensors should be distributed across GPUs")

  /** @group param */
  val grpAttnN = new IntParam(this, "grpAttnN", "Set the group-attention factor")

  /** @group param */
  val grpAttnW = new IntParam(this, "grpAttnW", "Set the group-attention width")

  /** @group param */
  val ropeFreqBase =
    new FloatParam(this, "ropeFreqBase", "Set the RoPE base frequency, used by NTK-aware scaling")

  /** @group param */
  val ropeFreqScale = new FloatParam(
    this,
    "ropeFreqScale",
    "Set the RoPE frequency scaling factor, expands context by a factor of 1/N")

  /** @group param */
  val yarnExtFactor =
    new FloatParam(this, "yarnExtFactor", "Set the YaRN extrapolation mix factor")

  /** @group param */
  val yarnAttnFactor =
    new FloatParam(this, "yarnAttnFactor", "Set the YaRN scale sqrt(t) or attention magnitude")

  /** @group param */
  val yarnBetaFast =
    new FloatParam(this, "yarnBetaFast", "Set the YaRN low correction dim or beta")

  /** @group param */
  val yarnBetaSlow =
    new FloatParam(this, "yarnBetaSlow", "Set the YaRN high correction dim or alpha")

  /** @group param */
  val yarnOrigCtx =
    new IntParam(this, "yarnOrigCtx", "Set the YaRN original context size of model")

  /** @group param */
  val defragmentationThreshold =
    new FloatParam(this, "defragmentationThreshold", "Set the KV cache defragmentation threshold")

  /** Set optimization strategies that help on some NUMA systems (if available)
    *
    * Available Strategies:
    *
    *   - DISABLED: No NUMA optimizations
    *   - DISTRIBUTE: Spread execution evenly over all
    *   - ISOLATE: Only spawn threads on CPUs on the node that execution started on
    *   - NUMA_CTL: Use the CPU map provided by numactl
    *   - MIRROR: Mirrors the model across NUMA nodes
    *
    * @group param
    */
  val numaStrategy = new Param[String](
    this,
    "numaStrategy",
    "Set optimization strategies that help on some NUMA systems (if available)")

  /** Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    *
    *   - UNSPECIFIED: Don't use any scaling
    *   - LINEAR: Linear scaling
    *   - YARN: YaRN RoPE scaling
    * @group param
    */
  val ropeScalingType = new Param[String](
    this,
    "ropeScalingType",
    "Set the RoPE frequency scaling method, defaults to linear unless specified by the model")

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - 0 UNSPECIFIED: Don't use any pooling
    *   - 1 MEAN: Mean Pooling
    *   - 2 CLS: CLS Pooling
    *
    * @group param
    */
  val poolingType = new Param[String](
    this,
    "poolingType",
    "Set the pooling type for embeddings, use model default if unspecified")
  //  model = new Param[String](this, "model", "Set the model file path to load")
  /** @group param */
  val modelDraft =
    new Param[String](this, "modelDraft", "Set the draft model for speculative decoding")

  //  modelAlias = new Param[String](this, "modelAlias", "Set a model alias")
  /** @group param */
  val lookupCacheStaticFilePath = new Param[String](
    this,
    "lookupCacheStaticFilePath",
    "Set path to static lookup cache to use for lookup decoding (not updated by generation)")

  /** @group param */
  val lookupCacheDynamicFilePath = new Param[String](
    this,
    "lookupCacheDynamicFilePath",
    "Set path to dynamic lookup cache to use for lookup decoding (updated by generation)")

  /** @group param */
  val loraAdapters = new StructFeature[Map[String, Float]](this, "loraAdapters")

  val embedding =
    new BooleanParam(this, "embedding", "Whether to load model with embedding support")

  /** @group param */
  val flashAttention =
    new BooleanParam(this, "flashAttention", "Whether to enable Flash Attention")

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
    new Param[String](this, "chatTemplate", "The chat template to use")

  /** Set the number of threads to use during generation
    *
    * @group setParam
    */
  def setNThreads(nThreads: Int): this.type = { set(this.nThreads, nThreads) }

  /** Set the number of threads to use during draft generation
    *
    * @group setParam
    */
  def setNThreadsDraft(nThreadsDraft: Int): this.type = { set(this.nThreadsDraft, nThreadsDraft) }

  /** Set the number of threads to use during batch and prompt processing
    *
    * @group setParam
    */
  def setNThreadsBatch(nThreadsBatch: Int): this.type = { set(this.nThreadsBatch, nThreadsBatch) }

  /** Set the number of threads to use during batch and prompt processing
    *
    * @group setParam
    */
  def setNThreadsBatchDraft(nThreadsBatchDraft: Int): this.type = {
    set(this.nThreadsBatchDraft, nThreadsBatchDraft)
  }

  /** Set the size of the prompt context
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

  /** Set the number of tokens to draft for speculative decoding
    *
    * @group setParam
    */
  def setNDraft(nDraft: Int): this.type = { set(this.nDraft, nDraft) }

  /** Set the maximal number of chunks to process
    *
    * @group setParam
    */
  def setNChunks(nChunks: Int): this.type = { set(this.nChunks, nChunks) }

  /** Set the number of sequences to decode
    *
    * @group setParam
    */
  def setNSequences(nSequences: Int): this.type = { set(this.nSequences, nSequences) }

  /** Set the speculative decoding split probability
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
  def setNGpuLayersDraft(nGpuLayersDraft: Int): this.type = {
    set(this.nGpuLayersDraft, nGpuLayersDraft)
  }

  /** Set how to split the model across GPUs
    *
    *   - NONE: No GPU split
    * -LAYER: Split the model across GPUs by layer 2. ROW: Split the model across GPUs by rows
    *
    * @group setParam
    */
  def setGpuSplitMode(splitMode: String): this.type = { set(this.gpuSplitMode, splitMode) }

  /** Set the GPU that is used for scratch and small tensors
    *
    * @group setParam
    */
  def setMainGpu(mainGpu: Int): this.type = { set(this.mainGpu, mainGpu) }

  /** Set how split tensors should be distributed across GPUs
    *
    * @group setParam
    */
  def setTensorSplit(tensorSplit: Array[Double]): this.type = {
    set(this.tensorSplit, tensorSplit)
  }

  /** Set the group-attention factor
    *
    * @group setParam
    */
  def setGrpAttnN(grpAttnN: Int): this.type = { set(this.grpAttnN, grpAttnN) }

  /** Set the group-attention width
    *
    * @group setParam
    */
  def setGrpAttnW(grpAttnW: Int): this.type = { set(this.grpAttnW, grpAttnW) }

  /** Set the RoPE base frequency, used by NTK-aware scaling
    *
    * @group setParam
    */
  def setRopeFreqBase(ropeFreqBase: Float): this.type = { set(this.ropeFreqBase, ropeFreqBase) }

  /** Set the RoPE frequency scaling factor, expands context by a factor of 1/N
    *
    * @group setParam
    */
  def setRopeFreqScale(ropeFreqScale: Float): this.type = {
    set(this.ropeFreqScale, ropeFreqScale)
  }

  /** Set the YaRN extrapolation mix factor
    *
    * @group setParam
    */
  def setYarnExtFactor(yarnExtFactor: Float): this.type = {
    set(this.yarnExtFactor, yarnExtFactor)
  }

  /** Set the YaRN scale sqrt(t) or attention magnitude
    *
    * @group setParam
    */
  def setYarnAttnFactor(yarnAttnFactor: Float): this.type = {
    set(this.yarnAttnFactor, yarnAttnFactor)
  }

  /** Set the YaRN low correction dim or beta
    *
    * @group setParam
    */
  def setYarnBetaFast(yarnBetaFast: Float): this.type = { set(this.yarnBetaFast, yarnBetaFast) }

  /** Set the YaRN high correction dim or alpha
    *
    * @group setParam
    */
  def setYarnBetaSlow(yarnBetaSlow: Float): this.type = { set(this.yarnBetaSlow, yarnBetaSlow) }

  /** Set the YaRN original context size of model
    *
    * @group setParam
    */
  def setYarnOrigCtx(yarnOrigCtx: Int): this.type = { set(this.yarnOrigCtx, yarnOrigCtx) }

  /** Set the KV cache defragmentation threshold
    *
    * @group setParam
    */
  def setDefragmentationThreshold(defragThold: Float): this.type = {
    set(this.defragmentationThreshold, defragThold)
  }

  /** Set optimization strategies that help on some NUMA systems (if available)
    *
    * Available Strategies:
    *
    *   - DISABLED: No NUMA optimizations
    *   - DISTRIBUTE: spread execution evenly over all
    *   - ISOLATE: only spawn threads on CPUs on the node that execution started on
    *   - NUMA_CTL: use the CPU map provided by numactl
    *   - MIRROR: Mirrors the model across NUMA nodes
    *
    * @group setParam
    */
  def setNumaStrategy(numa: String): this.type = { set(this.numaStrategy, numa) }

  /** Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    *
    *   - UNSPECIFIED: Don't use any scaling
    *   - LINEAR: Linear scaling
    *   - YARN: YaRN RoPE scaling
    * @group setParam
    */
  def setRopeScalingType(ropeScalingType: String): this.type = {
    set(this.ropeScalingType, ropeScalingType)
  }

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - UNSPECIFIED: Don't use any pooling
    *   - MEAN: Mean Pooling
    *   - CLS: CLS Pooling
    *
    * @group setParam
    */
  def setPoolingType(poolingType: String): this.type = { set(this.poolingType, poolingType) }

  /** Set the draft model for speculative decoding
    *
    * @group setParam
    */
  def setModelDraft(modelDraft: String): this.type = { set(this.modelDraft, modelDraft) }

  /** Set a model alias
    *
    * @group setParam
    */
  def setLookupCacheStaticFilePath(lookupCacheStaticFilePath: String): this.type = {
    set(this.lookupCacheStaticFilePath, lookupCacheStaticFilePath)
  }

  /** Set a model alias
    *
    * @group setParam
    */
  def setLookupCacheDynamicFilePath(lookupCacheDynamicFilePath: String): this.type = {
    set(this.lookupCacheDynamicFilePath, lookupCacheDynamicFilePath)
  }

  /** Sets paths to lora adapters with user defined scale.
    *
    * @group setParam
    */
  def setLoraAdapters(loraAdapters: Map[String, Float]): this.type = {
    set(this.loraAdapters, loraAdapters)
  }

  /** Sets paths to lora adapters with user defined scale. (PySpark Override)
    *
    * @group setParam
    */
  def setLoraAdapters(loraAdapters: java.util.HashMap[String, java.lang.Double]): this.type = {
    val scalaLoraAdapters = loraAdapters.asScala.map { case (k, v) => k -> v.floatValue() }
    set(this.loraAdapters, scalaLoraAdapters.toMap)
  }

  /** Whether to load model with embedding support
    *
    * @group setParam
    */
  def setEmbedding(embedding: Boolean): this.type = { set(this.embedding, embedding) }

  /** Whether to enable Flash Attention
    *
    * @group setParam
    */
  def setFlashAttention(flashAttention: Boolean): this.type = {
    set(this.flashAttention, flashAttention)
  }

  /** Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string
    *
    * @group setParam
    */
  def setInputPrefixBos(inputPrefixBos: Boolean): this.type = {
    set(this.inputPrefixBos, inputPrefixBos)
  }

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

  /** The chat template to use
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
  def getNSequences: Int = $(nSequences)

  /** @group getParam */
  def getPSplit: Float = $(pSplit)

  /** @group getParam */
  def getNGpuLayers: Int = $(nGpuLayers)

  /** @group getParam */
  def getNGpuLayersDraft: Int = $(nGpuLayersDraft)

  /** @group getParam */
  def getSplitMode: String = $(gpuSplitMode)

  /** @group getParam */
  def getMainGpu: Int = $(mainGpu)

  /** @group getParam */
  def getTensorSplit: Array[Double] = $(tensorSplit)

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
  def getDefragmentationThreshold: Float = $(defragmentationThreshold)

  /** @group getParam */
  def getNuma: String = $(numaStrategy)

  /** @group getParam */
  def getRopeScalingType: String = $(ropeScalingType)

  /** @group getParam */
  def getPoolingType: String = $(poolingType)

  /** @group getParam */
  def getModelDraft: String = $(modelDraft)

  /** @group getParam */
  def getLookupCacheStaticFilePath: String = $(lookupCacheStaticFilePath)

  /** @group getParam */
  def getLookupCacheDynamicFilePath: String = $(lookupCacheDynamicFilePath)

  /** @group getParam */
  def getLoraAdapters: Map[String, Float] = $$(loraAdapters)

  /** @group getParam */
  def getEmbedding: Boolean = $(embedding)

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
  /** @group param */
  val inputPrefix =
    new Param[String](this, "inputPrefix", "Set the prompt to start generation with")

  /** @group param */
  val inputSuffix =
    new Param[String](this, "inputSuffix", "Set a suffix for infilling")

  /** @group param */
  val cachePrompt = new BooleanParam(
    this,
    "cachePrompt",
    "Whether to remember the prompt to avoid reprocessing it")

  /** @group param */
  val nPredict = new IntParam(this, "nPredict", "Set the number of tokens to predict")

  /** @group param */
  val topK = new IntParam(this, "topK", "Set top-k sampling")

  /** @group param */
  val topP = new FloatParam(this, "topP", "Set top-p sampling")

  /** @group param */
  val minP = new FloatParam(this, "minP", "Set min-p sampling")

  /** @group param */
  val tfsZ = new FloatParam(this, "tfsZ", "Set tail free sampling, parameter z")

  /** @group param */
  val typicalP = new FloatParam(this, "typicalP", "Set locally typical sampling, parameter p")

  /** @group param */
  val temperature = new FloatParam(this, "temperature", "Set the temperature")

  /** @group param */
  val dynamicTemperatureRange =
    new FloatParam(this, "dynatempRange", "Set the dynamic temperature range")

  /** @group param */
  val dynamicTemperatureExponent =
    new FloatParam(this, "dynatempExponent", "Set the dynamic temperature exponent")

  /** @group param */
  val repeatLastN =
    new IntParam(this, "repeatLastN", "Set the last n tokens to consider for penalties")

  /** @group param */
  val repeatPenalty =
    new FloatParam(this, "repeatPenalty", "Set the penalty of repeated sequences of tokens")

  /** @group param */
  val frequencyPenalty =
    new FloatParam(this, "frequencyPenalty", "Set the repetition alpha frequency penalty")

  /** @group param */
  val presencePenalty =
    new FloatParam(this, "presencePenalty", "Set the repetition alpha presence penalty")

  /** @group param */
  val miroStat = new Param[String](this, "miroStat", "Set MiroStat sampling strategies.")

  /** @group param */
  val miroStatTau =
    new FloatParam(this, "mirostatTau", "Set the MiroStat target entropy, parameter tau")

  /** @group param */
  val miroStatEta =
    new FloatParam(this, "mirostatEta", "Set the MiroStat learning rate, parameter eta")

  /** @group param */
  val penalizeNl = new BooleanParam(this, "penalizeNl", "Whether to penalize newline tokens")

  /** @group param */
  val nKeep =
    new IntParam(this, "nKeep", "Set the number of tokens to keep from the initial prompt")

  /** @group param */
  val seed = new IntParam(this, "seed", "Set the RNG seed")

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
  val tokenIdBias: StructFeature[Map[Int, Float]] =
    new StructFeature[Map[Int, Float]](this, "tokenIdBias")

  // Modify the likelihood of tokens appearing in the completion by their string.
  /** @group param */
  val tokenBias: StructFeature[Map[String, Float]] =
    new StructFeature[Map[String, Float]](this, "tokenBias")

  /** @group param */
  val disableTokenIds =
    new IntArrayParam(this, "disableTokenIds", "Set the token ids to disable in the completion")

  /** @group param */
  val stopStrings = new StringArrayParam(
    this,
    "stopStrings",
    "Set strings upon seeing which token generation is stopped")

  /** @group param */
  val samplers = new StringArrayParam(
    this,
    "samplers",
    "Set which samplers to use for token generation in the given order")

  /** @group param */
  val useChatTemplate = new BooleanParam(
    this,
    "useChatTemplate",
    "Set whether or not generate should apply a chat template")

  /** Set the prompt to start generation with
    *
    * @group setParam
    */
  def setInputPrefix(inputPrefix: String): this.type = { set(this.inputPrefix, inputPrefix) }

  /** Set a suffix for infilling
    *
    * @group setParam
    */
  def setInputSuffix(inputSuffix: String): this.type = { set(this.inputSuffix, inputSuffix) }

  /** Whether to remember the prompt to avoid reprocessing it
    *
    * @group setParam
    */
  def setCachePrompt(cachePrompt: Boolean): this.type = { set(this.cachePrompt, cachePrompt) }

  /** Set the number of tokens to predict
    *
    * @group setParam
    */
  def setNPredict(nPredict: Int): this.type = { set(this.nPredict, nPredict) }

  /** Set top-k sampling
    *
    * @group setParam
    */
  def setTopK(topK: Int): this.type = { set(this.topK, topK) }

  /** Set top-p sampling
    *
    * @group setParam
    */
  def setTopP(topP: Float): this.type = { set(this.topP, topP) }

  /** Set min-p sampling
    *
    * @group setParam
    */
  def setMinP(minP: Float): this.type = { set(this.minP, minP) }

  /** Set tail free sampling, parameter z
    * @group setParam
    */
  def setTfsZ(tfsZ: Float): this.type = { set(this.tfsZ, tfsZ) }

  /** Set locally typical sampling, parameter p
    *
    * @group setParam
    */
  def setTypicalP(typicalP: Float): this.type = { set(this.typicalP, typicalP) }

  /** Set the temperature
    *
    * @group setParam
    */
  def setTemperature(temperature: Float): this.type = { set(this.temperature, temperature) }

  /** Set the dynamic temperature range
    *
    * @group setParam
    */
  def setDynamicTemperatureRange(dynatempRange: Float): this.type = {
    set(this.dynamicTemperatureRange, dynatempRange)
  }

  /** Set the dynamic temperature exponent
    *
    * @group setParam
    */
  def setDynamicTemperatureExponent(dynatempExponent: Float): this.type = {
    set(this.dynamicTemperatureExponent, dynatempExponent)
  }

  /** Set the last n tokens to consider for penalties
    *
    * @group setParam
    */
  def setRepeatLastN(repeatLastN: Int): this.type = { set(this.repeatLastN, repeatLastN) }

  /** Set the penalty of repeated sequences of tokens
    *
    * @group setParam
    */
  def setRepeatPenalty(repeatPenalty: Float): this.type = {
    set(this.repeatPenalty, repeatPenalty)
  }

  /** Set the repetition alpha frequency penalty
    *
    * @group setParam
    */
  def setFrequencyPenalty(frequencyPenalty: Float): this.type = {
    set(this.frequencyPenalty, frequencyPenalty)
  }

  /** Set the repetition alpha presence penalty
    *
    * @group setParam
    */
  def setPresencePenalty(presencePenalty: Float): this.type = {
    set(this.presencePenalty, presencePenalty)
  }

  /** Set MiroStat sampling strategies.
    *
    *   - DISABLED: No MiroStat
    *   - V1: MiroStat V1
    *   - V2: MiroStat V2
    *
    * @group setParam
    */
  def setMiroStat(mirostat: String): this.type = set(this.miroStat, mirostat)

  /** Set the MiroStat target entropy, parameter tau
    *
    * @group setParam
    */
  def setMiroStatTau(mirostatTau: Float): this.type = { set(this.miroStatTau, mirostatTau) }

  /** Set the MiroStat learning rate, parameter eta
    *
    * @group setParam
    */
  def setMiroStatEta(mirostatEta: Float): this.type = { set(this.miroStatEta, mirostatEta) }

  /** Set whether to penalize newline tokens
    *
    * @group setParam
    */
  def setPenalizeNl(penalizeNl: Boolean): this.type = { set(this.penalizeNl, penalizeNl) }

  /** Set the number of tokens to keep from the initial prompt
    *
    * @group setParam
    */
  def setNKeep(nKeep: Int): this.type = { set(this.nKeep, nKeep) }

  /** Set the RNG seed
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
  def setPenaltyPrompt(penaltyPrompt: String): this.type = {
    set(this.penaltyPrompt, penaltyPrompt)
  }

  /** Set whether to ignore end of stream token and continue generating (implies --logit-bias
    * 2-inf)
    *
    * @group setParam
    */
  def setIgnoreEos(ignoreEos: Boolean): this.type = { set(this.ignoreEos, ignoreEos) }

  /** Set the tokens to disable during completion.
    *
    * @group setParam
    */
  def setTokenBias(tokenBias: Map[String, Float]): this.type = {
    set(this.tokenBias, tokenBias)
  }

  /** Set the tokens to disable during completion. (Override for PySpark)
    *
    * @group setParam
    */
  def setTokenBias(tokenBias: java.util.HashMap[String, java.lang.Double]): this.type = {
    val scalaTokenBias = tokenBias.asScala.map { case (k, v) => k -> v.floatValue() }
    set(this.tokenBias, scalaTokenBias.toMap)
  }

  /** Set the token ids to disable in the completion.
    *
    * @group setParam
    */
  def setTokenIdBias(tokenIdBias: Map[Int, Float]): this.type = {
    set(this.tokenIdBias, tokenIdBias)
  }

  /** Set the token ids to disable in the completion. (Override for PySpark)
    *
    * @group setParam
    */
  def setTokenIdBias(tokenIdBias: java.util.HashMap[Integer, java.lang.Double]): this.type = {
    val scalaTokenIdBias = tokenIdBias.asScala.map { case (k, v) => k.toInt -> v.toFloat }
    set(this.tokenIdBias, scalaTokenIdBias.toMap)
  }

  /** Set the token ids to disable in the completion. This corresponds to `setTokenBias` with a
    * value of `Float.NEGATIVE_INFINITY`.
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
  def setStopStrings(stopStrings: Array[String]): this.type = {
    set(this.stopStrings, stopStrings)
  }

  /** Set which samplers to use for token generation in the given order .
    *
    * Available Samplers are:
    *
    *   - TOP_K: Top-k sampling
    *   - TFS_Z: Tail free sampling
    *   - TYPICAL_P: Locally typical sampling p
    *   - TOP_P: Top-p sampling
    *   - MIN_P: Min-p sampling
    *   - TEMPERATURE: Temperature sampling
    * @group setParam
    */
  def setSamplers(samplers: Array[String]): this.type = { set(this.samplers, samplers) }

  /** Set whether or not generate should apply a chat template
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
  def getMiroStat: String = $(miroStat)

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
  def getTokenIdBias: Map[Int, Float] = $$(tokenIdBias)

  /** @group getParam */
  def getTokenBias: Map[String, Float] = $$(tokenBias)

  /** @group getParam */
  def getDisableTokenIds: Array[Int] = $(disableTokenIds)

  /** @group getParam */
  def getStopStrings: Array[String] = $(stopStrings)

  /** @group getParam */
  def getSamplers: Array[String] = $(samplers)

  /** @group getParam */
  def getUseChatTemplate: Boolean = $(useChatTemplate)

  protected def getModelParameters: ModelParameters = {
    val modelParameters = new ModelParameters().setContinuousBatching(true) // Always enabled

    if (isDefined(chatTemplate)) modelParameters.setChatTemplate($(chatTemplate))
    if (isDefined(defragmentationThreshold))
      modelParameters.setDefragmentationThreshold($(defragmentationThreshold))
    if (isDefined(embedding)) modelParameters.setEmbedding($(embedding))
    if (isDefined(flashAttention)) modelParameters.setFlashAttention($(flashAttention))
    if (isDefined(gpuSplitMode))
      modelParameters.setSplitMode(GpuSplitMode.valueOf($(gpuSplitMode)))
    if (isDefined(grpAttnN)) modelParameters.setGrpAttnN($(grpAttnN))
    if (isDefined(grpAttnW)) modelParameters.setGrpAttnN($(grpAttnW))
    if (isDefined(inputPrefixBos)) modelParameters.setInputPrefixBos($(inputPrefixBos))
    if (isDefined(lookupCacheDynamicFilePath))
      modelParameters.setLookupCacheDynamicFilePath($(lookupCacheDynamicFilePath))
    if (isDefined(lookupCacheStaticFilePath))
      modelParameters.setLookupCacheStaticFilePath($(lookupCacheStaticFilePath))
    if (isDefined(mainGpu)) modelParameters.setMainGpu($(mainGpu))
    if (isDefined(modelDraft)) modelParameters.setModelDraft($(modelDraft))
    if (isDefined(nBatch)) modelParameters.setNBatch($(nBatch))
    if (isDefined(nChunks)) modelParameters.setNChunks($(nChunks))
    if (isDefined(nCtx)) modelParameters.setNCtx($(nCtx))
    if (isDefined(nDraft)) modelParameters.setNDraft($(nDraft))
    if (isDefined(nGpuLayers)) modelParameters.setNGpuLayers($(nGpuLayers))
    if (isDefined(nGpuLayersDraft)) modelParameters.setNGpuLayersDraft($(nGpuLayersDraft))
    if (isDefined(nSequences)) modelParameters.setNSequences($(nSequences))
    if (isDefined(nThreads)) modelParameters.setNThreads($(nThreads))
    if (isDefined(nThreadsBatch)) modelParameters.setNThreadsBatch($(nThreadsBatch))
    if (isDefined(nThreadsBatchDraft))
      modelParameters.setNThreadsBatchDraft($(nThreadsBatchDraft))
    if (isDefined(nThreadsDraft)) modelParameters.setNThreadsDraft($(nThreadsDraft))
    if (isDefined(nUbatch)) modelParameters.setNUbatch($(nUbatch))
    if (isDefined(noKvOffload)) modelParameters.setNoKvOffload($(noKvOffload))
    if (isDefined(numaStrategy)) modelParameters.setNuma(NumaStrategy.valueOf($(numaStrategy)))
    if (isDefined(pSplit)) modelParameters.setPSplit($(pSplit))
    if (isDefined(poolingType))
      modelParameters.setPoolingType(PoolingType.valueOf($(poolingType)))
    if (isDefined(ropeFreqBase)) modelParameters.setRopeFreqBase($(ropeFreqBase))
    if (isDefined(ropeFreqScale)) modelParameters.setRopeFreqScale($(ropeFreqScale))
    if (isDefined(ropeScalingType))
      modelParameters.setRopeScalingType(RopeScalingType.valueOf($(ropeScalingType)))
    if (isDefined(systemPrompt)) modelParameters.setSystemPrompt($(systemPrompt))
    if (isDefined(tensorSplit)) modelParameters.setTensorSplit($(tensorSplit).map(_.toFloat))
    if (isDefined(useMlock)) modelParameters.setUseMlock($(useMlock))
    if (isDefined(useMmap)) modelParameters.setUseMmap($(useMmap))
    if (isDefined(yarnAttnFactor)) modelParameters.setYarnAttnFactor($(yarnAttnFactor))
    if (isDefined(yarnBetaFast)) modelParameters.setYarnBetaFast($(yarnBetaFast))
    if (isDefined(yarnBetaSlow)) modelParameters.setYarnBetaSlow($(yarnBetaSlow))
    if (isDefined(yarnExtFactor)) modelParameters.setYarnExtFactor($(yarnExtFactor))
    if (isDefined(yarnOrigCtx)) modelParameters.setYarnOrigCtx($(yarnOrigCtx))
    if (loraAdapters.isSet) {
      val loraAdaptersMap: mutable.Map[String, java.lang.Float] =
        mutable.Map($$(loraAdapters).map { case (key, value) =>
          (key, float2Float(value))
        }.toSeq: _*)
      modelParameters.setLoraAdapters(loraAdaptersMap.asJava)
    } // Need to convert to mutable map first

    modelParameters
  }

  protected def getInferenceParameters: InferenceParameters = {
    val inferenceParams = new InferenceParameters("")
    if (isDefined(cachePrompt)) inferenceParams.setCachePrompt($(cachePrompt))
    if (isDefined(disableTokenIds)) {
      val javaCollection: java.util.Collection[Integer] =
        $(disableTokenIds).map(int2Integer).toSeq.asJava
      inferenceParams.disableTokenIds(javaCollection)
    }
    if (isDefined(dynamicTemperatureExponent))
      inferenceParams.setDynamicTemperatureExponent($(dynamicTemperatureExponent))
    if (isDefined(dynamicTemperatureRange))
      inferenceParams.setDynamicTemperatureRange($(dynamicTemperatureRange))
    if (isDefined(frequencyPenalty)) inferenceParams.setFrequencyPenalty($(frequencyPenalty))
    if (isDefined(grammar)) inferenceParams.setGrammar($(grammar))
    if (isDefined(ignoreEos)) inferenceParams.setIgnoreEos($(ignoreEos))
    if (isDefined(inputPrefix)) inferenceParams.setInputPrefix($(inputPrefix))
    if (isDefined(inputSuffix)) inferenceParams.setInputSuffix($(inputSuffix))
    if (isDefined(minKeep)) inferenceParams.setMinKeep($(minKeep))
    if (isDefined(minP)) inferenceParams.setMinP($(minP))
    if (isDefined(miroStat)) inferenceParams.setMiroStat(MiroStat.valueOf($(miroStat)))
    if (isDefined(miroStatEta)) inferenceParams.setMiroStatEta($(miroStatEta))
    if (isDefined(miroStatTau)) inferenceParams.setMiroStatTau($(miroStatTau))
    if (isDefined(nKeep)) inferenceParams.setNKeep($(nKeep))
    if (isDefined(nPredict)) inferenceParams.setNPredict($(nPredict))
    if (isDefined(nProbs)) inferenceParams.setNProbs($(nProbs))
    if (isDefined(penalizeNl)) inferenceParams.setPenalizeNl($(penalizeNl))
    if (isDefined(penaltyPrompt)) inferenceParams.setPenaltyPrompt($(penaltyPrompt))
    if (isDefined(presencePenalty)) inferenceParams.setPresencePenalty($(presencePenalty))
    if (isDefined(repeatLastN)) inferenceParams.setRepeatLastN($(repeatLastN))
    if (isDefined(repeatPenalty)) inferenceParams.setRepeatPenalty($(repeatPenalty))
    if (isDefined(samplers)) inferenceParams.setSamplers($(samplers).map(Sampler.valueOf): _*)
    if (isDefined(seed)) inferenceParams.setSeed($(seed))
    if (isDefined(stopStrings)) inferenceParams.setStopStrings($(stopStrings): _*)
    if (isDefined(temperature)) inferenceParams.setTemperature($(temperature))
    if (isDefined(tfsZ)) inferenceParams.setTfsZ($(tfsZ))
    if (isDefined(topK)) inferenceParams.setTopK($(topK))
    if (isDefined(topP)) inferenceParams.setTopP($(topP))
    if (isDefined(typicalP)) inferenceParams.setTypicalP($(typicalP))
    if (isDefined(useChatTemplate)) inferenceParams.setUseChatTemplate($(useChatTemplate))
    if (tokenBias.isSet) {
      val tokenBiasMap: mutable.Map[String, java.lang.Float] = mutable.Map($$(tokenBias).map {
        case (key, value) => (key, float2Float(value))
      }.toSeq: _*)
      inferenceParams.setTokenBias(tokenBiasMap.asJava)
    }
    if (tokenIdBias.isSet) {
      val tokenIdBiasMap: mutable.Map[Integer, java.lang.Float] =
        mutable.Map($$(tokenIdBias).map { case (key, value) =>
          (int2Integer(key), float2Float(value))
        }.toSeq: _*)
      inferenceParams.setTokenIdBias(tokenIdBiasMap.asJava)
    }

    inferenceParams
  }

  // ---------------- METADATA ----------------
  val metadata =
    new Param[String](this, "metadata", "Set the metadata for the model").setProtected()

  /** Set the metadata for the model
    * @group setParam
    */
  def setMetadata(metadata: String): this.type = { set(this.metadata, metadata) }

  /** Get the metadata for the model
    * @group getParam
    */
  def getMetadata: String = $(metadata)
}
