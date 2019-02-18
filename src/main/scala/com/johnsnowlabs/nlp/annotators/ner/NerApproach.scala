package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.param.{IntParam, Param, Params, StringArrayParam}

trait NerApproach[T <: NerApproach[_]] extends Params {
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")
  val entities = new StringArrayParam(this, "entities", "Entities to recognize")

  val minEpochs = new IntParam(this, "minEpochs", "Minimum number of epochs to train")
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")

  val randomSeed = new IntParam(this, "randomSeed", "Random seed")
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  def setLabelColumn(column: String) = set(labelColumn, column).asInstanceOf[T]
  def setEntities(tags: Array[String]) = set(entities, tags).asInstanceOf[T]
  def setMinEpochs(epochs: Int) = set(minEpochs, epochs).asInstanceOf[T]
  def setMaxEpochs(epochs: Int) = set(maxEpochs, epochs).asInstanceOf[T]
  def setVerbose(verbose: Int) = set(this.verbose, verbose).asInstanceOf[T]
  def setVerbose(verbose: Verbose.Level) = set(this.verbose, verbose.id).asInstanceOf[T]
  def setRandomSeed(seed: Int) = set(randomSeed, seed).asInstanceOf[T]
}

object Verbose extends Enumeration {
  type Level = Value

  val All = Value(0)
  val PerStep = Value(1)
  val Epochs = Value(2)
  val TrainingStat = Value(3)
  val Silent = Value(4)
}