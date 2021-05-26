package com.johnsnowlabs.nlp.annotators.spell.symmetric

import org.apache.spark.ml.param.{IntParam, LongParam, Params}

/**
  *
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
  */
trait SymmetricDeleteParams extends Params {

  /** max edit distance characters to derive strings from a word
    *
    * @group param
    **/
  val maxEditDistance = new IntParam(this, "maxEditDistance", "max edit distance characters to derive strings from a word")
  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.
    *
    * @group param
    **/
  val frequencyThreshold = new IntParam(this, "frequencyThreshold", "minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.")
  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0
    *
    * @group param
    **/
  val deletesThreshold = new IntParam(this, "deletesThreshold", "minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0")
  /** maximum duplicate of characters in a word to consider. Defaults to 2
    *
    * @group param
    **/
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  /** length of longest word in corpus
    *
    * @group param
    **/
  val longestWordLength = new IntParam(this, "longestWordLength", "length of longest word in corpus")
  /** minimum frequency of a word in the corpus
    *
    * @group param
    **/
  val minFrequency = new LongParam(this, "minFrequency", "minimum frequency of a word in the corpus")
  /** maximum frequency of a word in the corpus
    *
    * @group param
    **/
  val maxFrequency = new LongParam(this, "maxFrequency", "maximum frequency of a word in the corpus")

  /** max edit distance characters to derive strings from a word
    *
    * @group setParam
    **/
  def setMaxEditDistance(value: Int): this.type = set(maxEditDistance, value)

  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.
    *
    * @group setParam
    **/
  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)

  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0
    *
    * @group setParam
    **/
  def setDeletesThreshold(value: Int): this.type = set(deletesThreshold, value)

  /** maximum duplicate of characters in a word to consider. Defaults to 2 */
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)

  /** length of longest word in corpus
    *
    * @group setParam
    **/
  def setLongestWordLength(value: Int): this.type = set(longestWordLength, value)

  /** maximum frequency of a word in the corpus
    *
    * @group setParam
    **/
  def setMaxFrequency(value: Long): this.type = set(maxFrequency, value)

  /** minimum frequency of a word in the corpus
    *
    * @group setParam
    **/
  def setMinFrequency(value: Long): this.type = set(minFrequency, value)

  /** max edit distance characters to derive strings from a word
    *
    * @group getParam
    **/
  def getMaxEditDistance: Int = $(maxEditDistance)

  /** minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0.
    *
    * @group getParam
    **/
  def getFrequencyThreshold: Int = $(frequencyThreshold)

  /** minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0
    *
    * @group getParam
    **/
  def getDeletesThreshold: Int = $(deletesThreshold)

  /** maximum duplicate of characters in a word to consider. Defaults to 2
    *
    * @group getParam
    **/
  def getDupsLimit: Int = $(dupsLimit)

}
