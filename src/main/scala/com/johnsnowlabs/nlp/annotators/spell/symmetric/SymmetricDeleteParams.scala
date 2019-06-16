package com.johnsnowlabs.nlp.annotators.spell.symmetric

import org.apache.spark.ml.param.{IntParam, Params}

trait SymmetricDeleteParams extends Params{

  val maxEditDistance = new IntParam(this, "maxEditDistance",
                               "max edit distance characters to derive strings from a word")
  val frequencyThreshold = new IntParam(this, "frequencyThreshold",
    "minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.")
  val deletesThreshold = new IntParam(this, "deletesThreshold",
    "minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0")
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")

  def setMaxEditDistance(value: Int): this.type = set(maxEditDistance, value)
  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)
  def setDeletesThreshold(value: Int): this.type = set(deletesThreshold, value)
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)

  def getFrequencyThreshold: Int = $(frequencyThreshold)
  def getDeletesThreshold: Int = $(deletesThreshold)

}
