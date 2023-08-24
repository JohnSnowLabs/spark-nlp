package com.johnsnowlabs.nlp

import org.apache.spark.ml.param._

/** Parameters to configure beam search text generation. */
trait HasGeneratorProperties {
  this: ParamsAndFeaturesWritable =>

  /** Set transformer task, e.g. `"summarize:"` (Default: `""`).
    *
    * @group param
    */
  val task = new Param[String](this, "task", "Set transformer task, e.g. 'summarize'")

  /** @group setParam */
  def setTask(value: String): this.type = {
    if (get(task).isEmpty)
      set(task, value)
    this
  }

  /** @group getParam */
  def getTask: Option[String] = get(this.task)

  /** max length of the input sequence (Default: `0`)
    *
    * @group param
    */
  val maxInputLength =
    new IntParam(this, "maxInputLength", "Maximum length of the input sequence")

  def setMaxInputLength(value: Int): this.type = {
    set(maxInputLength, value)
    this
  }

  /** Minimum length of the sequence to be generated (Default: `0`)
    *
    * @group param
    */
  val minOutputLength =
    new IntParam(this, "minOutputLength", "Minimum length of the sequence to be generated")

  /** @group setParam */
  def setMinOutputLength(value: Int): this.type = {
    set(minOutputLength, value)
    this
  }

  /** @group getParam */
  def getMinOutputLength: Int = $(this.minOutputLength)

  /** Maximum length of the sequence to be generated (Default: `20`)
    *
    * @group param
    */
  val maxOutputLength =
    new IntParam(this, "maxOutputLength", "Maximum length of the sequence to be generated")

  /** @group setParam */
  def setMaxOutputLength(value: Int): this.type = {
    set(maxOutputLength, value)
    this
  }

  /** @group getParam */
  def getMaxOutputLength: Int = $(this.maxOutputLength)

  /** Whether or not to use sampling, use greedy decoding otherwise (Default: `false`)
    *
    * @group param
    */
  val doSample = new BooleanParam(
    this,
    "doSample",
    "Whether or not to use sampling; use greedy decoding otherwise")

  /** @group setParam */
  def setDoSample(value: Boolean): this.type = {
    set(doSample, value)
    this
  }

  /** @group getParam */
  def getDoSample: Boolean = $(this.doSample)

  /** The value used to module the next token probabilities (Default: `1.0`)
    *
    * @group param
    */
  val temperature =
    new DoubleParam(this, "temperature", "The value used to module the next token probabilities")

  /** @group setParam */
  def setTemperature(value: Double): this.type = {
    set(temperature, value)
    this
  }

  /** @group getParam */
  def getTemperature: Double = $(this.temperature)

  /** The number of highest probability vocabulary tokens to keep for top-k-filtering (Default:
    * `50`)
    *
    * @group param
    */
  val topK = new IntParam(
    this,
    "topK",
    "The number of highest probability vocabulary tokens to keep for top-k-filtering")

  /** @group setParam */
  def setTopK(value: Int): this.type = {
    set(topK, value)
    this
  }

  /** @group getParam */
  def getTopK: Int = $(this.topK)

  /** If set to float < `1.0`, only the most probable tokens with probabilities that add up to
    * `topP` or higher are kept for generation (Default: `1.0`)
    *
    * @group param
    */
  val topP = new DoubleParam(
    this,
    "topP",
    "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation")

  /** @group setParam */
  def setTopP(value: Double): this.type = {
    set(topP, value)
    this
  }

  /** @group getParam */
  def getTopP: Double = $(this.topP)

  /** The parameter for repetition penalty (Default: `1.0`). `1.0` means no penalty. See
    * [[https://arxiv.org/pdf/1909.05858.pdf this paper]] for more details.
    *
    * @group param
    */
  val repetitionPenalty = new DoubleParam(
    this,
    "repetitionPenalty",
    "The parameter for repetition penalty. 1.0 means no penalty.")

  /** @group setParam */
  def setRepetitionPenalty(value: Double): this.type = {
    set(repetitionPenalty, value)
    this
  }

  /** @group getParam */
  def getRepetitionPenalty: Double = $(this.repetitionPenalty)

  /** If set to int > `0`, all ngrams of that size can only occur once (Default: `0`)
    *
    * @group param
    */
  val noRepeatNgramSize = new IntParam(
    this,
    "noRepeatNgramSize",
    "If set to int > 0, all ngrams of that size can only occur once")

  /** @group setParam */
  def setNoRepeatNgramSize(value: Int): this.type = {
    set(noRepeatNgramSize, value)
    this
  }

  /** @group getParam */
  def getNoRepeatNgramSize: Int = $(this.noRepeatNgramSize)

  /** Optional Random seed for the model. Needs to be of type `Int`.
    *
    * @group param
    */
  var randomSeed: Option[Long] = None

  /** @group setParam */
  def setRandomSeed(value: Long): this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** @group getParam */
  def getRandomSeed: Option[Long] = this.randomSeed

  /** Beam size for the beam search algorithm (Default: `4`)
    *
    * @group param
    */
  val beamSize = new IntParam(this, "beamSize", "Number of beams for beam search.")

  /** @group setParam */
  def setBeamSize(beamNum: Int): this.type = {
    set(beamSize, beamNum)
  }

  /** @group getParam */
  def getBeamSize: Int = $(beamSize)

  /** The number of sequences to return from the beam search.
    *
    * @group param
    */
  val nReturnSequences = new IntParam(
    this,
    "nReturnSequences",
    "The number of sequences to return from the beam search.")

  /** @group setParam */
  def setNReturnSequences(beamNum: Int): this.type = {
    set(nReturnSequences, beamNum)
  }

  /** @group getParam */
  def getNReturnSequences: Int = $(nReturnSequences)
}
