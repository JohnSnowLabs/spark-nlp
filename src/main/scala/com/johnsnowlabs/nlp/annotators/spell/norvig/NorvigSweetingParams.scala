package com.johnsnowlabs.nlp.annotators.spell.norvig

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params}


/** These are the configs for the NorvigSweeting model
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala]] for further reference on how to use this API
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
  **/
trait NorvigSweetingParams extends Params {


  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy
    *
    * @group param
    **/
  val caseSensitive = new BooleanParam(this, "caseSensitive", "sensitivity on spell checking")
  /** Increase search at cost of performance. Enables extra check for word combinations, More accuracy at performance
    *
    * @group param
    **/
  val doubleVariants = new BooleanParam(this, "doubleVariants", "increase search at cost of performance")
  /** Increase performance at cost of accuracy. Faster but less accurate mode
    *
    * @group param
    **/
  val shortCircuit = new BooleanParam(this, "shortCircuit", "increase performance at cost of accuracy")
  /** Applies frequency over hamming in intersections. When false hamming takes priority
    *
    * @group param
    **/
  val frequencyPriority = new BooleanParam(this, "frequencyPriority", "applies frequency over hamming in intersections. When false hamming takes priority")
  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3.
    *
    * @group param
    **/
  val wordSizeIgnore = new IntParam(this, "wordSizeIgnore", "minimum size of word before ignoring. Defaults to 3")
  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2.
    *
    * @group param
    **/
  val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  /** Word reduction limit. Defaults to 3
    *
    * @group param
    **/
  val reductLimit = new IntParam(this, "reductLimit", "word reductions limit. Defaults to 3")
  /** Hamming intersections to attempt. Defaults to 10
    *
    * @group param
    **/
  val intersections = new IntParam(this, "intersections", "hamming intersections to attempt. Defaults to 10")
  /** Vowel swap attempts. Defaults to 6
    *
    * @group param
    **/
  val vowelSwapLimit = new IntParam(this, "vowelSwapLimit", "vowel swap attempts. Defaults to 6")

  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy
    *
    * @group setParam
    **/
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** Increase search at cost of performance. Enables extra check for word combinations
    *
    * @group setParam
    **/
  def setDoubleVariants(value: Boolean): this.type = set(doubleVariants, value)

  /** Increase performance at cost of accuracy. Faster but less accurate mode
    *
    * @group setParam
    **/
  def setShortCircuit(value: Boolean): this.type = set(shortCircuit, value)

  /** Applies frequency over hamming in intersections. When false hamming takes priority
    *
    * @group setParam
    **/
  def setFrequencyPriority(value: Boolean): this.type = set(frequencyPriority, value)

  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3.
    *
    * @group setParam
    **/
  def setWordSizeIgnore(value: Int): this.type = set(wordSizeIgnore, value)

  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2. */
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)

  /** Word reduction limit. Defaults to 3
    *
    * @group setParam
    **/
  def setReductLimit(value: Int): this.type = set(reductLimit, value)

  /** Hamming intersections to attempt. Defaults to 10
    *
    * @group setParam
    **/
  def setIntersections(value: Int): this.type = set(intersections, value)

  /** Vowel swap attempts. Defaults to 6
    *
    * @group setParam
    **/
  def setVowelSwapLimit(value: Int): this.type = set(vowelSwapLimit, value)

  /** Sensitivity on spell checking. Defaults to false. Might affect accuracy
    *
    * @group getParam
    **/
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Increase search at cost of performance. Enables extra check for word combinations
    *
    * @group getParam
    **/
  def getDoubleVariants: Boolean = $(doubleVariants)

  /** Increase performance at cost of accuracy. Faster but less accurate mode
    *
    * @group getParam
    **/
  def getShortCircuit: Boolean = $(shortCircuit)

  /** Applies frequency over hamming in intersections. When false hamming takes priority
    *
    * @group getParam
    **/
  def getFrequencyPriority: Boolean = $(frequencyPriority)

  /** Minimum size of word before ignoring. Defaults to 3 ,Minimum size of word before moving on. Defaults to 3.
    *
    * @group getParam
    **/
  def getWordSizeIgnore: Int = $(wordSizeIgnore)

  /** Maximum duplicate of characters in a word to consider. Defaults to 2 .Maximum duplicate of characters to account for. Defaults to 2.
    *
    * @group getParam
    **/
  def getDupsLimit: Int = $(dupsLimit)

  /** Word reduction limit. Defaults to 3
    *
    * @group getParam
    **/
  def getReductLimit: Int = $(reductLimit)

  /** Hamming intersections to attempt. Defaults to 10
    *
    * @group getParam
    **/
  def getIntersections: Int = $(intersections)

  /** Vowel swap attempts. Defaults to 6
    *
    * @group getParam
    **/
  def getVowelSwapLimit: Int = $(vowelSwapLimit)

}
