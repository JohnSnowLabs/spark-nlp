package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
import de.kherud.llama.InferenceParameters
import de.kherud.llama.args._
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param._

import scala.collection.mutable
import scala.jdk.CollectionConverters._

/** Contains settable inference parameters for the [[AutoGGUFModel]].
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
trait HasLlamaCppInferenceProperties {
  this: ParamsAndFeaturesWritable with HasProtectedParams =>

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

  protected def getInferenceParameters: InferenceParameters = {
    val inferenceParams = new InferenceParameters("")
    if (isDefined(cachePrompt)) inferenceParams.setCachePrompt(getCachePrompt)
    if (isDefined(disableTokenIds)) {
      val javaCollection: java.util.Collection[Integer] =
        getDisableTokenIds.map(int2Integer).toSeq.asJava
      inferenceParams.disableTokenIds(javaCollection)
    }
    if (isDefined(dynamicTemperatureExponent))
      inferenceParams.setDynamicTemperatureExponent(getDynamicTemperatureExponent)
    if (isDefined(dynamicTemperatureRange))
      inferenceParams.setDynamicTemperatureRange(getDynamicTemperatureRange)
    if (isDefined(frequencyPenalty)) inferenceParams.setFrequencyPenalty(getFrequencyPenalty)
    if (isDefined(grammar)) inferenceParams.setGrammar(getGrammar)
    if (isDefined(ignoreEos)) inferenceParams.setIgnoreEos(getIgnoreEos)
    if (isDefined(inputPrefix)) inferenceParams.setInputPrefix(getInputPrefix)
    if (isDefined(inputSuffix)) inferenceParams.setInputSuffix(getInputSuffix)
    if (isDefined(minKeep)) inferenceParams.setMinKeep(getMinKeep)
    if (isDefined(minP)) inferenceParams.setMinP(getMinP)
    if (isDefined(miroStat)) inferenceParams.setMiroStat(MiroStat.valueOf(getMiroStat))
    if (isDefined(miroStatEta)) inferenceParams.setMiroStatEta(getMiroStatEta)
    if (isDefined(miroStatTau)) inferenceParams.setMiroStatTau(getMiroStatTau)
    if (isDefined(nKeep)) inferenceParams.setNKeep(getNKeep)
    if (isDefined(nPredict)) inferenceParams.setNPredict(getNPredict)
    if (isDefined(nProbs)) inferenceParams.setNProbs(getNProbs)
    if (isDefined(penalizeNl)) inferenceParams.setPenalizeNl(getPenalizeNl)
    if (isDefined(penaltyPrompt)) inferenceParams.setPenaltyPrompt(getPenaltyPrompt)
    if (isDefined(presencePenalty)) inferenceParams.setPresencePenalty(getPresencePenalty)
    if (isDefined(repeatLastN)) inferenceParams.setRepeatLastN(getRepeatLastN)
    if (isDefined(repeatPenalty)) inferenceParams.setRepeatPenalty(getRepeatPenalty)
    if (isDefined(samplers)) inferenceParams.setSamplers(getSamplers.map(Sampler.valueOf): _*)
    if (isDefined(seed)) inferenceParams.setSeed(getSeed)
    if (isDefined(stopStrings)) inferenceParams.setStopStrings(getStopStrings: _*)
    if (isDefined(temperature)) inferenceParams.setTemperature(getTemperature)
    if (isDefined(tfsZ)) inferenceParams.setTfsZ(getTfsZ)
    if (isDefined(topK)) inferenceParams.setTopK(getTopK)
    if (isDefined(topP)) inferenceParams.setTopP(getTopP)
    if (isDefined(typicalP)) inferenceParams.setTypicalP(getTypicalP)
    if (isDefined(useChatTemplate)) inferenceParams.setUseChatTemplate(getUseChatTemplate)
    if (tokenBias.isSet) {
      val tokenBiasMap: mutable.Map[String, java.lang.Float] = mutable.Map(getTokenBias.map {
        case (key, value) => (key, float2Float(value))
      }.toSeq: _*)
      inferenceParams.setTokenBias(tokenBiasMap.asJava)
    }
    if (tokenIdBias.isSet) {
      val tokenIdBiasMap: mutable.Map[Integer, java.lang.Float] =
        mutable.Map(getTokenIdBias.map { case (key, value) =>
          (int2Integer(key), float2Float(value))
        }.toSeq: _*)
      inferenceParams.setTokenIdBias(tokenIdBiasMap.asJava)
    }

    inferenceParams
  }

}
