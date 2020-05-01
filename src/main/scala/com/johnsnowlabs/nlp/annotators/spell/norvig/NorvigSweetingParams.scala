package com.johnsnowlabs.nlp.annotators.spell.norvig

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params}


/** These are the configs for the NorvigSweeting model
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala]] for further reference on how to use this API
  * */
trait NorvigSweetingParams extends Params {


  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy */
  val caseSensitive = new BooleanParam(this, "caseSensitive", "sensitivity on spell checking")
  /** Increase search at cost of performance. Enables extra check for word combinations, More accuracy at performance */
  val doubleVariants = new BooleanParam(this, "doubleVariants", "increase search at cost of performance")
  /** Increase performance at cost of accuracy. Faster but less accurate mode */
  val shortCircuit = new BooleanParam(this, "shortCircuit", "increase performance at cost of accuracy")
  /** Applies frequency over hamming in intersections. When false hamming takes priority */
  val frequencyPriority = new BooleanParam(this, "frequencyPriority",
    "applies frequency over hamming in intersections. When false hamming takes priority")

  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3. */
  val wordSizeIgnore = new IntParam(this, "wordSizeIgnore", "minimum size of word before ignoring. Defaults to 3")
  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2. */
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  /** Word reduction limit. Defaults to 3 */
  val reductLimit = new IntParam(this, "reductLimit", "word reductions limit. Defaults to 3")
  /** Hamming intersections to attempt. Defaults to 10 */
  val intersections = new IntParam(this, "intersections", "hamming intersections to attempt. Defaults to 10")
  /** Vowel swap attempts. Defaults to 6 */
  val vowelSwapLimit = new IntParam(this, "vowelSwapLimit", "vowel swap attempts. Defaults to 6")

  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** Increase search at cost of performance. Enables extra check for word combinations */
  def setDoubleVariants(value: Boolean): this.type = set(doubleVariants, value)

  /** Increase performance at cost of accuracy. Faster but less accurate mode */
  def setShortCircuit(value: Boolean): this.type = set(shortCircuit, value)

  /** Applies frequency over hamming in intersections. When false hamming takes priority */
  def setFrequencyPriority(value: Boolean): this.type = set(frequencyPriority, value)

  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3. */
  def setWordSizeIgnore(value: Int): this.type = set(wordSizeIgnore, value)

  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2. */
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)

  /** Word reduction limit. Defaults to 3 */
  def setReductLimit(value: Int): this.type = set(reductLimit, value)

  /** Hamming intersections to attempt. Defaults to 10 */
  def setIntersections(value: Int): this.type = set(intersections, value)

  /** Vowel swap attempts. Defaults to 6 */
  def setVowelSwapLimit(value: Int): this.type = set(vowelSwapLimit, value)

  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Increase search at cost of performance. Enables extra check for word combinations */
  def getDoubleVariants: Boolean = $(doubleVariants)

  /** Increase performance at cost of accuracy. Faster but less accurate mode */
  def getShortCircuit: Boolean = $(shortCircuit)

  /** Applies frequency over hamming in intersections. When false hamming takes priority */
  def getFrequencyPriority: Boolean = $(frequencyPriority)

  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3. */
  def getWordSizeIgnore: Int = $(wordSizeIgnore)

  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2. */
  def getDupsLimit: Int = $(dupsLimit)

  /** Word reduction limit. Defaults to 3 */
  def getReductLimit: Int = $(reductLimit)

  /** Hamming intersections to attempt. Defaults to 10 */
  def getIntersections: Int = $(intersections)

  /** Vowel swap attempts. Defaults to 6 */
  def getVowelSwapLimit: Int = $(vowelSwapLimit)

}
