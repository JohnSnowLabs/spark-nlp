package com.johnsnowlabs.nlp.annotators.spell.symmetric

import org.apache.spark.ml.param.{IntParam, LongParam, Params}

trait SymmetricDeleteParams extends Params{

  /** max edit distance characters to derive strings from a word */
  val maxEditDistance = new IntParam(this, "maxEditDistance", "max edit distance characters to derive strings from a word")
  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0. */
  val frequencyThreshold = new IntParam(this, "frequencyThreshold", "minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.")
  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0 */
  val deletesThreshold = new IntParam(this, "deletesThreshold", "minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0")
  /** maximum duplicate of characters in a word to consider. Defaults to 2 */
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  /** length of longest word in corpus */
  val longestWordLength = new IntParam(this, "longestWordLength", "length of longest word in corpus")
  /** minimum frequency of a word in the corpus */
  val minFrequency = new LongParam(this, "minFrequency", "minimum frequency of a word in the corpus")
  /** maximum frequency of a word in the corpus */
  val maxFrequency = new LongParam(this, "maxFrequency", "maximum frequency of a word in the corpus")

  /** max edit distance characters to derive strings from a word */
  def setMaxEditDistance(value: Int): this.type = set(maxEditDistance, value)

  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0. */
  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)

  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0 */
  def setDeletesThreshold(value: Int): this.type = set(deletesThreshold, value)

  /** maximum duplicate of characters in a word to consider. Defaults to 2 */
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)

  /** length of longest word in corpus */
  def setLongestWordLength(value: Int): this.type = set(longestWordLength, value)

  /** maximum frequency of a word in the corpus */
  def setMaxFrequency(value: Long): this.type = set(maxFrequency, value)

  /** minimum frequency of a word in the corpus */
  def setMinFrequency(value: Long): this.type = set(minFrequency, value)

  /** max edit distance characters to derive strings from a word */
  def getMaxEditDistance: Int = $(maxEditDistance)

  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0. */
  def getFrequencyThreshold: Int = $(frequencyThreshold)

  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0 */
  def getDeletesThreshold: Int = $(deletesThreshold)

  /** maximum duplicate of characters in a word to consider. Defaults to 2 */
  def getDupsLimit: Int = $(dupsLimit)

}
