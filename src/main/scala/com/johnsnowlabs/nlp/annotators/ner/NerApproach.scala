package com.johnsnowlabs.nlp.annotators.ner

import org.apache.spark.ml.param.{IntParam, Param, Params, StringArrayParam}

trait NerApproach[T <: NerApproach[_]] extends Params {

  /** Column with label per each token */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")
  /** Entities to recognize */
  val entities = new StringArrayParam(this, "entities", "Entities to recognize")
  /** Minimum number of epochs to train */
  val minEpochs = new IntParam(this, "minEpochs", "Minimum number of epochs to train")
  /** Maximum number of epochs to train */
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  /** Random seed */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")
  /** Level of verbosity during training */
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** Column with label per each token */
  def setLabelColumn(column: String): T = set(labelColumn, column).asInstanceOf[T]

  /** Entities to recognize */
  def setEntities(tags: Array[String]): T = set(entities, tags).asInstanceOf[T]

  /** Minimum number of epochs to train */
  def setMinEpochs(epochs: Int): T = set(minEpochs, epochs).asInstanceOf[T]

  /** Maximum number of epochs to train */
  def setMaxEpochs(epochs: Int): T = set(maxEpochs, epochs).asInstanceOf[T]

  /** Level of verbosity during training  */
  def setVerbose(verbose: Int): T = set(this.verbose, verbose).asInstanceOf[T]

  /** Level of verbosity during training  */
  def setVerbose(verbose: Verbose.Level): T = set(this.verbose, verbose.id).asInstanceOf[T]

  /** Random seed  */
  def setRandomSeed(seed: Int): T = set(randomSeed, seed).asInstanceOf[T]

  /** Minimum number of epochs to train */
  def getMinEpochs: Int = $(minEpochs)

  /** Maximum number of epochs to train */
  def getMaxEpochs: Int = $(maxEpochs)

  /** Level of verbosity during training  */
  def getVerbose: Int = $(verbose)

  /** Random seed  */
  def getRandomSeed: Int = $(randomSeed)

}

object Verbose extends Enumeration {
  type Level = Value

  val All = Value(0)
  val PerStep = Value(1)
  val Epochs = Value(2)
  val TrainingStat = Value(3)
  val Silent = Value(4)
}