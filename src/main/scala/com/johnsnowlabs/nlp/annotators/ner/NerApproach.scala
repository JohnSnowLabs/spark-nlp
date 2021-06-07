package com.johnsnowlabs.nlp.annotators.ner

import org.apache.spark.ml.param.{IntParam, Param, Params, StringArrayParam}

/**
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
trait NerApproach[T <: NerApproach[_]] extends Params {

  /** Column with label per each token
    *
    * @group param
    **/
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")
  /** Entities to recognize
    *
    * @group param
    **/
  val entities = new StringArrayParam(this, "entities", "Entities to recognize")
  /** Minimum number of epochs to train
    *
    * @group param
    **/
  val minEpochs = new IntParam(this, "minEpochs", "Minimum number of epochs to train")
  /** Maximum number of epochs to train
    *
    * @group param
    **/
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  /** Random seed
    *
    * @group param
    **/
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")
  /** Level of verbosity during training
    *
    * @group param
    **/
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** Column with label per each token
    *
    * @group setParam
    **/
  def setLabelColumn(column: String): T = set(labelColumn, column).asInstanceOf[T]

  /** Entities to recognize
    *
    * @group setParam
    **/
  def setEntities(tags: Array[String]): T = set(entities, tags).asInstanceOf[T]

  /** Minimum number of epochs to train
    *
    * @group setParam
    **/
  def setMinEpochs(epochs: Int): T = set(minEpochs, epochs).asInstanceOf[T]

  /** Maximum number of epochs to train
    *
    * @group setParam
    **/
  def setMaxEpochs(epochs: Int): T = set(maxEpochs, epochs).asInstanceOf[T]

  /** Level of verbosity during training
    *
    * @group setParam
    **/
  def setVerbose(verbose: Int): T = set(this.verbose, verbose).asInstanceOf[T]

  /** Level of verbosity during training
    *
    * @group setParam
    **/
  def setVerbose(verbose: Verbose.Level): T = set(this.verbose, verbose.id).asInstanceOf[T]

  /** Random seed
    *
    * @group setParam
    **/
  def setRandomSeed(seed: Int): T = set(randomSeed, seed).asInstanceOf[T]

  /** Minimum number of epochs to train
    *
    * @group getParam
    **/
  def getMinEpochs: Int = $(minEpochs)

  /** Maximum number of epochs to train
    *
    * @group getParam
    **/
  def getMaxEpochs: Int = $(maxEpochs)

  /** Level of verbosity during training
    *
    * @group getParam
    **/
  def getVerbose: Int = $(verbose)

  /** Random seed
    *
    * @group getParam
    **/
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