package com.johnsnowlabs.nlp.annotators.spell.norvig

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params}

trait NorvigSweetingParams extends Params {


  val caseSensitive = new BooleanParam(this, "caseSensitive", "sensitivity on spell checking")
  val doubleVariants = new BooleanParam(this, "doubleVariants", "increase search at cost of performance")
  val shortCircuit = new BooleanParam(this, "shortCircuit", "increase performance at cost of accuracy")
  val frequencyPriority = new BooleanParam(this, "frequencyPriority",
    "applies frequency over hamming in intersections. When false hamming takes priority")

  val wordSizeIgnore = new IntParam(this, "wordSizeIgnore", "minimum size of word before ignoring. Defaults to 3")
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  val reductLimit = new IntParam(this, "reductLimit", "word reductions limit. Defaults to 3")
  val intersections = new IntParam(this, "intersections", "hamming intersections to attempt. Defaults to 10")
  val vowelSwapLimit = new IntParam(this, "vowelSwapLimit", "vowel swap attempts. Defaults to 6")

  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)
  def setDoubleVariants(value: Boolean): this.type = set(doubleVariants, value)
  def setShortCircuit(value: Boolean): this.type = set(shortCircuit, value)
  def setFrequencyPriority(value: Boolean): this.type = set(frequencyPriority, value)

  def setWordSizeIgnore(value: Int): this.type = set(wordSizeIgnore, value)
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)
  def setReductLimit(value: Int): this.type = set(reductLimit, value)
  def setIntersections(value: Int): this.type = set(intersections, value)
  def setVowelSwapLimit(value: Int): this.type = set(vowelSwapLimit, value)

}
