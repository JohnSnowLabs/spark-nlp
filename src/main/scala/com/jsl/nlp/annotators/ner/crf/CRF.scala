package com.jsl.nlp.annotators.ner.crf

import org.apache.spark.rdd.RDD

/**
 * CRF with support for multiple parallel runs
 * L2 regParam = 1/(2.0 * sigma**2)
 */
class CRF private (
  private var freq: Int,
  private var regParam: Double,
  private var maxIterations: Int,
  private var tolerance: Double,
  private var regularization: String
) extends Serializable {

  def this() = this(freq = 1, regParam = 0.5, maxIterations = 1000, tolerance = 1E-3, regularization = "L2")

  def setRegParam(regParam: Double) = {
    this.regParam = regParam
    this
  }

  def setFreq(freq: Int) = {
    this.freq = freq
    this
  }

  def setMaxIterations(maxIterations: Int) = {
    this.maxIterations = maxIterations
    this
  }

  def setEta(eta: Double) = {
    this.tolerance = eta
    this
  }

  def setRegularization(regula: String) = {
    this.regularization = regula
    this
  }

  /**
   * Internal method to train the CRF model
   *
   * @param template the template to train the model
   * @param trains the source for the training
   * @return the model of the source
   */
  def runCRF(
    template: Array[String],
    trains: RDD[Sequence]
  ): CRFModel = {
    val featureIdx = new FeatureIndex()
    featureIdx.openTemplate(template)
    featureIdx.openTagSetDist(trains)

    val bcFeatureIdxI = trains.context.broadcast(featureIdx)
    val taggers = trains
      .map(new Tagger(bcFeatureIdxI.value.labels.size, LearnMode).read(_, bcFeatureIdxI.value))

    featureIdx.buildDictionaryDist(taggers, bcFeatureIdxI, freq)

    val bcFeatureIdxII = trains.context.broadcast(featureIdx)
    val taggerList: RDD[Tagger] = taggers.map(bcFeatureIdxII.value.buildFeatures(_)).cache()

    val model = runAlgorithm(taggerList, featureIdx)
    taggerList.unpersist()

    model
  }

  /**
   *
   * @param taggers the tagger in the template
   * @param featureIdx the index of the feature
   */
  def runAlgorithm(
    taggers: RDD[Tagger],
    featureIdx: FeatureIndex
  ): CRFModel = {

    println("Starting CRF Iterations ( sentences: %d, features: %d, labels: %d )"
      .format(taggers.count(), featureIdx.maxID, featureIdx.labels.length))

    var updater: UpdaterCRF = null
    regularization match {
      case "L1" =>
        updater = new L1Updater
      case "L2" =>
        updater = new L2Updater
      case _ =>
        throw new Exception("only support L1-CRF and L2-CRF now")
    }

    featureIdx.alpha = new CRFWithLBFGS(new CRFGradient, updater)
      .setRegParam(regParam)
      .setConvergenceTol(tolerance)
      .setNumIterations(maxIterations)
      .optimizer(taggers, featureIdx.initAlpha())

    featureIdx.saveModel
  }
}

/**
 * Top-level methods for calling CRF.
 */
object CRF {

  /**
   * Train CRF Model
   *
   * @param templates Source templates for training the model
   * @param train Source files for training the model
   * @return Model
   */

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double,
    freq: Int,
    maxIteration: Int,
    eta: Double,
    regularization: String
  ): CRFModel = {
    new CRF().setRegParam(regParam)
      .setFreq(freq)
      .setMaxIterations(maxIteration)
      .setEta(eta)
      .setRegularization(regularization)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double,
    freq: Int,
    maxIteration: Int,
    eta: Double
  ): CRFModel = {
    new CRF().setRegParam(regParam)
      .setFreq(freq)
      .setMaxIterations(maxIteration)
      .setEta(eta)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double,
    freq: Int
  ): CRFModel = {
    new CRF().setRegParam(regParam)
      .setFreq(freq)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double,
    regularization: String
  ): CRFModel = {
    new CRF().setRegParam(regParam)
      .setRegularization(regularization)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regularization: String
  ): CRFModel = {
    new CRF().setRegularization(regularization)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence]
  ): CRFModel = {
    new CRF().runCRF(templates, train)
  }
}
