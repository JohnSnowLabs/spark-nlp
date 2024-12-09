package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
import com.johnsnowlabs.nlp.llama.ModelParameters
import com.johnsnowlabs.nlp.llama.args.{GpuSplitMode, NumaStrategy, PoolingType, RopeScalingType}
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param._
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.jdk.CollectionConverters._

/** Contains settable model parameters for the [[AutoGGUFModel]].
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
trait HasLlamaCppModelProperties {
  this: ParamsAndFeaturesWritable with HasProtectedParams =>
  protected val logger = LoggerFactory.getLogger(this.getClass)

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
    *
    * @group param
    */
  val ropeScalingType = new Param[String](
    this,
    "ropeScalingType",
    "Set the RoPE frequency scaling method, defaults to linear unless specified by the model")

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - 0 NONE: Don't use any pooling
    *   - 1 MEAN: Mean Pooling
    *   - 2 CLS: Choose the CLS token
    *   - 3 LAST: Choose the last token
    *
    * @group param
    */
  val poolingType = new Param[String](
    this,
    "poolingType",
    "Set the pooling type for embeddings, use model default if unspecified")

  /** @group param */
  val modelDraft =
    new Param[String](this, "modelDraft", "Set the draft model for speculative decoding")

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

  /** @group param */
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

  private def checkEmbeddingMode(setter: => this.type): this.type = {
    if (getEmbedding) {
      logger.warn("Embeddings enabled. This parameter has no effect.")
      this
    } else
      setter
  }

  /** Set the number of threads to use during generation
    *
    * @group setParam
    */
  def setNThreads(nThreads: Int): this.type = {
    set(this.nThreads, nThreads)
  }

  /** Set the number of threads to use during draft generation
    *
    * @group setParam
    */
  def setNThreadsDraft(nThreadsDraft: Int): this.type = {
    checkEmbeddingMode { set(this.nThreadsDraft, nThreadsDraft) }
  }

  /** Set the number of threads to use during batch and prompt processing
    *
    * @group setParam
    */
  def setNThreadsBatch(nThreadsBatch: Int): this.type = {
    checkEmbeddingMode { set(this.nThreadsBatch, nThreadsBatch) }
  }

  /** Set the number of threads to use during batch and prompt processing
    *
    * @group setParam
    */
  def setNThreadsBatchDraft(nThreadsBatchDraft: Int): this.type = {
    checkEmbeddingMode { set(this.nThreadsBatchDraft, nThreadsBatchDraft) }
  }

  /** Set the size of the prompt context
    *
    * @group setParam
    */
  def setNCtx(nCtx: Int): this.type = {
    set(this.nCtx, nCtx)
  }

  /** Set the logical batch size for prompt processing (must be >=32 to use BLAS)
    *
    * @group setParam
    */
  def setNBatch(nBatch: Int): this.type = {
    set(this.nBatch, nBatch)
  }

  /** Set the physical batch size for prompt processing (must be >=32 to use BLAS)
    *
    * @group setParam
    */
  def setNUbatch(nUbatch: Int): this.type = {
    set(this.nUbatch, nUbatch)
  }

  /** Set the number of tokens to draft for speculative decoding
    *
    * @group setParam
    */
  def setNDraft(nDraft: Int): this.type = {
    checkEmbeddingMode { set(this.nDraft, nDraft) }
  }

  /** Set the maximal number of chunks to process
    *
    * @group setParam
    */
  def setNChunks(nChunks: Int): this.type = {
    set(this.nChunks, nChunks)
  }

  /** Set the number of sequences to decode
    *
    * @group setParam
    */
  def setNSequences(nSequences: Int): this.type = {
    set(this.nSequences, nSequences)
  }

  /** Set the speculative decoding split probability
    *
    * @group setParam
    */
  def setPSplit(pSplit: Float): this.type = {
    checkEmbeddingMode { set(this.pSplit, pSplit) }
  }

  /** Set the number of layers to store in VRAM (-1 - use default)
    *
    * @group setParam
    */
  def setNGpuLayers(nGpuLayers: Int): this.type = {
    set(this.nGpuLayers, nGpuLayers)
  }

  /** Set the number of layers to store in VRAM for the draft model (-1 - use default)
    *
    * @group setParam
    */
  def setNGpuLayersDraft(nGpuLayersDraft: Int): this.type = {
    checkEmbeddingMode { set(this.nGpuLayersDraft, nGpuLayersDraft) }
  }

  /** Set how to split the model across GPUs
    *
    *   - NONE: No GPU split
    * -LAYER: Split the model across GPUs by layer 2. ROW: Split the model across GPUs by rows
    *
    * @group setParam
    */
  def setGpuSplitMode(splitMode: String): this.type = {
    set(this.gpuSplitMode, splitMode)
  }

  /** Set the GPU that is used for scratch and small tensors
    *
    * @group setParam
    */
  def setMainGpu(mainGpu: Int): this.type = {
    set(this.mainGpu, mainGpu)
  }

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
  def setGrpAttnN(grpAttnN: Int): this.type = {
    set(this.grpAttnN, grpAttnN)
  }

  /** Set the group-attention width
    *
    * @group setParam
    */
  def setGrpAttnW(grpAttnW: Int): this.type = {
    set(this.grpAttnW, grpAttnW)
  }

  /** Set the RoPE base frequency, used by NTK-aware scaling
    *
    * @group setParam
    */
  def setRopeFreqBase(ropeFreqBase: Float): this.type = {
    set(this.ropeFreqBase, ropeFreqBase)
  }

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
  def setYarnBetaFast(yarnBetaFast: Float): this.type = {
    set(this.yarnBetaFast, yarnBetaFast)
  }

  /** Set the YaRN high correction dim or alpha
    *
    * @group setParam
    */
  def setYarnBetaSlow(yarnBetaSlow: Float): this.type = {
    set(this.yarnBetaSlow, yarnBetaSlow)
  }

  /** Set the YaRN original context size of model
    *
    * @group setParam
    */
  def setYarnOrigCtx(yarnOrigCtx: Int): this.type = {
    set(this.yarnOrigCtx, yarnOrigCtx)
  }

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
  def setNumaStrategy(numa: String): this.type = {
    val numaUpper = numa.toUpperCase
    val numaStrategies = Array("DISABLED", "DISTRIBUTE", "ISOLATE", "NUMA_CTL", "MIRROR")
    require(
      numaStrategies.contains(numaUpper),
      s"Invalid NUMA strategy: $numa. " +
        s"Valid values are: ${numaStrategies.mkString(", ")}")
    set(this.numaStrategy, numaUpper)
  }

  /** Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    *
    *   - UNSPECIFIED: Don't use any scaling
    *   - LINEAR: Linear scaling
    *   - YARN: YaRN RoPE scaling
    *
    * @group setParam
    */
  def setRopeScalingType(ropeScalingType: String): this.type = {
    set(this.ropeScalingType, ropeScalingType)
  }

  /** Set the pooling type for embeddings, use model default if unspecified
    *
    *   - 0 NONE: Don't use any pooling and return token embeddings (if the model supports it)
    *   - 1 MEAN: Mean Pooling
    *   - 2 CLS: Choose the CLS token
    *   - 3 LAST: Choose the last token
    *
    * @group setParam
    */
  def setPoolingType(poolingType: String): this.type = {
    val poolingTypeUpper = poolingType.toUpperCase
    val poolingTypes = Array("NONE", "MEAN", "CLS", "LAST")
    require(
      poolingTypes.contains(poolingTypeUpper),
      s"Invalid pooling type: $poolingType. " +
        s"Valid values are: ${poolingTypes.mkString(", ")}")
    set(this.poolingType, poolingTypeUpper)
  }

  /** Set the draft model for speculative decoding
    *
    * @group setParam
    */
  def setModelDraft(modelDraft: String): this.type = {
    checkEmbeddingMode { set(this.modelDraft, modelDraft) }
  }

  /** Set path to static lookup cache to use for lookup decoding (not updated by generation)
    *
    * @group setParam
    */
  def setLookupCacheStaticFilePath(lookupCacheStaticFilePath: String): this.type = {
    checkEmbeddingMode { set(this.lookupCacheStaticFilePath, lookupCacheStaticFilePath) }
  }

  /** Set path to dynamic lookup cache to use for lookup decoding (updated by generation)
    *
    * @group setParam
    */
  def setLookupCacheDynamicFilePath(lookupCacheDynamicFilePath: String): this.type = {
    checkEmbeddingMode { set(this.lookupCacheDynamicFilePath, lookupCacheDynamicFilePath) }
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
  def setEmbedding(embedding: Boolean): this.type = {
    set(this.embedding, embedding)
  }

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
  def setUseMmap(useMmap: Boolean): this.type = {
    set(this.useMmap, useMmap)
  }

  /** Whether to force the system to keep model in RAM rather than swapping or compressing
    *
    * @group setParam
    */
  def setUseMlock(useMlock: Boolean): this.type = {
    set(this.useMlock, useMlock)
  }

  /** Whether to disable KV offload
    *
    * @group setParam
    */
  def setNoKvOffload(noKvOffload: Boolean): this.type = {
    set(this.noKvOffload, noKvOffload)
  }

  /** Set a system prompt to use
    *
    * @group setParam
    */
  def setSystemPrompt(systemPrompt: String): this.type = {
    checkEmbeddingMode { set(this.systemPrompt, systemPrompt) }
  }

  /** The chat template to use
    *
    * @group setParam
    */
  def setChatTemplate(chatTemplate: String): this.type = {
    checkEmbeddingMode { set(this.chatTemplate, chatTemplate) }
  }

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

  def getMetadataMap: Map[String, String] = {
    val metadataJsonString = getMetadata
    if (metadataJsonString.isEmpty) Map.empty
    else {
      implicit val formats: DefaultFormats.type = DefaultFormats
      JsonMethods.parse(metadataJsonString).extract[Map[String, String]]
    }
  }

  protected def getModelParameters: ModelParameters = {
    val modelParameters = new ModelParameters().setContinuousBatching(true) // Always enabled

    if (isDefined(chatTemplate)) modelParameters.setChatTemplate(getChatTemplate)
    if (isDefined(defragmentationThreshold))
      modelParameters.setDefragmentationThreshold(getDefragmentationThreshold)
    if (isDefined(embedding)) modelParameters.setEmbedding(getEmbedding)
    if (isDefined(flashAttention)) modelParameters.setFlashAttention(getFlashAttention)
    if (isDefined(gpuSplitMode))
      modelParameters.setSplitMode(GpuSplitMode.valueOf(getSplitMode))
    if (isDefined(grpAttnN)) modelParameters.setGrpAttnN(getGrpAttnN)
    if (isDefined(grpAttnW)) modelParameters.setGrpAttnN(getGrpAttnW)
    if (isDefined(inputPrefixBos)) modelParameters.setInputPrefixBos(getInputPrefixBos)
    if (isDefined(lookupCacheDynamicFilePath))
      modelParameters.setLookupCacheDynamicFilePath(getLookupCacheDynamicFilePath)
    if (isDefined(lookupCacheStaticFilePath))
      modelParameters.setLookupCacheStaticFilePath(getLookupCacheStaticFilePath)
    if (isDefined(mainGpu)) modelParameters.setMainGpu(getMainGpu)
    if (isDefined(modelDraft)) modelParameters.setModelDraft(getModelDraft)
    if (isDefined(nBatch)) modelParameters.setNBatch(getNBatch)
    if (isDefined(nChunks)) modelParameters.setNChunks(getNChunks)
    if (isDefined(nCtx)) modelParameters.setNCtx(getNCtx)
    if (isDefined(nDraft)) modelParameters.setNDraft(getNDraft)
    if (isDefined(nGpuLayers)) modelParameters.setNGpuLayers(getNGpuLayers)
    if (isDefined(nGpuLayersDraft)) modelParameters.setNGpuLayersDraft(getNGpuLayersDraft)
    if (isDefined(nSequences)) modelParameters.setNSequences(getNSequences)
    if (isDefined(nThreads)) modelParameters.setNThreads(getNThreads)
    if (isDefined(nThreadsBatch)) modelParameters.setNThreadsBatch(getNThreadsBatch)
    if (isDefined(nThreadsBatchDraft))
      modelParameters.setNThreadsBatchDraft(getNThreadsBatchDraft)
    if (isDefined(nThreadsDraft)) modelParameters.setNThreadsDraft(getNThreadsDraft)
    if (isDefined(nUbatch)) modelParameters.setNUbatch(getNUbatch)
    if (isDefined(noKvOffload)) modelParameters.setNoKvOffload(getNoKvOffload)
    if (isDefined(numaStrategy)) modelParameters.setNuma(NumaStrategy.valueOf(getNuma))
    if (isDefined(pSplit)) modelParameters.setPSplit(getPSplit)
    if (isDefined(poolingType))
      modelParameters.setPoolingType(PoolingType.valueOf(getPoolingType))
    if (isDefined(ropeFreqBase)) modelParameters.setRopeFreqBase(getRopeFreqBase)
    if (isDefined(ropeFreqScale)) modelParameters.setRopeFreqScale(getRopeFreqScale)
    if (isDefined(ropeScalingType))
      modelParameters.setRopeScalingType(RopeScalingType.valueOf(getRopeScalingType))
    if (isDefined(systemPrompt)) modelParameters.setSystemPrompt(getSystemPrompt)
    if (isDefined(tensorSplit)) modelParameters.setTensorSplit(getTensorSplit.map(_.toFloat))
    if (isDefined(useMlock)) modelParameters.setUseMlock(getUseMlock)
    if (isDefined(useMmap)) modelParameters.setUseMmap(getUseMmap)
    if (isDefined(yarnAttnFactor)) modelParameters.setYarnAttnFactor(getYarnAttnFactor)
    if (isDefined(yarnBetaFast)) modelParameters.setYarnBetaFast(getYarnBetaFast)
    if (isDefined(yarnBetaSlow)) modelParameters.setYarnBetaSlow(getYarnBetaSlow)
    if (isDefined(yarnExtFactor)) modelParameters.setYarnExtFactor(getYarnExtFactor)
    if (isDefined(yarnOrigCtx)) modelParameters.setYarnOrigCtx(getYarnOrigCtx)
    if (loraAdapters.isSet) {
      val loraAdaptersMap: mutable.Map[String, java.lang.Float] =
        mutable.Map(getLoraAdapters.map { case (key, value) =>
          (key, float2Float(value))
        }.toSeq: _*)
      modelParameters.setLoraAdapters(loraAdaptersMap.asJava)
    } // Need to convert to mutable map first

    modelParameters
  }

  // ---------------- GPU SUPPORT ----------------
  // Values for automatic GPU support
  protected val defaultGpuLayers = 1000
  protected val defaultMainGpu = 0

  // Entrypoint for models. Automatically set GPU support if detected.
  protected def setGpuSupportIfAvailable(spark: SparkSession): this.type = {
    val usingGPUJar: Boolean = spark.sparkContext.listJars.exists(_.contains("spark-nlp-gpu"))
    if (usingGPUJar) {
      logger.info("Using GPU jar. Offloading all layers to GPU.")
      setMainGpu(defaultMainGpu)
      setNGpuLayers(defaultGpuLayers)
    }
    this
  }
}
