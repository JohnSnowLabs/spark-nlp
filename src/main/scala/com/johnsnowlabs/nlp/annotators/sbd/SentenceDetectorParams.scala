package com.johnsnowlabs.nlp.annotators.sbd

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params, StringArrayParam}

import scala.collection.mutable.ArrayBuffer

trait SentenceDetectorParams extends Params {

  val useAbbrevations = new BooleanParam(this, "useAbbreviations", "whether to apply abbreviations at sentence detection")

  val useCustomBoundsOnly = new BooleanParam(this, "useCustomBoundsOnly", "whether to only utilize custom bounds for sentence detection")

  val explodeSentences = new BooleanParam(this, "explodeSentences", "whether to explode each sentence into a different row, for better parallelization. Defaults to false.")

  val customBounds: StringArrayParam = new StringArrayParam(
    this,
    "customBounds",
    "characters used to explicitly mark sentence bounds"
  )

  val splitLength: IntParam = new IntParam(this, "splitLength", "length at which sentences will be forcibly split.")
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each sentence")
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each sentence")

  setDefault(
    useAbbrevations -> true,
    useCustomBoundsOnly -> false,
    explodeSentences -> false,
    customBounds -> Array.empty[String],
    minLength -> 0
  )

  def setCustomBounds(value: Array[String]): this.type = set(customBounds, value)

  def getCustomBounds: Array[String] = $(customBounds)

  def setUseCustomBoundsOnly(value: Boolean): this.type = set(useCustomBoundsOnly, value)

  def getUseCustomBoundsOnly: Boolean = $(useCustomBoundsOnly)

  def setUseAbbreviations(value: Boolean): this.type = set(useAbbrevations, value)

  def getUseAbbreviations: Boolean = $(useAbbrevations)

  def setExplodeSentences(value: Boolean): this.type = set(explodeSentences, value)

  def getExplodeSentences: Boolean = $(explodeSentences)

  def setSplitLength(value: Int): this.type = set(splitLength, value)

  def getSplitLength: Int = $(splitLength)

  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }
  def getMinLength(value: Int): Int = $(minLength)

  def setMaxLength(value: Int): this.type = {
    require(value >= ${minLength}, "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }
  def getMaxLength(value: Int): Int = $(maxLength)


  def truncateSentence(sentence: String, maxLength: Int): Array[String] = {
    var currentLength = 0
    val allSentences = ArrayBuffer.empty[String]
    val currentSentence = ArrayBuffer.empty[String]

    def addWordToSentence(word: String): Unit = {
      /** Adds +1 because of the space joining words */
      currentLength += word.length + 1
      currentSentence.append(word)
    }

    sentence.split(" ").foreach(word => {
      if (currentLength + word.length > maxLength) {
        allSentences.append(currentSentence.mkString(" "))
        currentSentence.clear()
        currentLength = 0
        addWordToSentence(word)
      }
      else {
        addWordToSentence(word)
      }
    })
    /** add leftovers */
    allSentences.append(currentSentence.mkString(" "))
    allSentences.toArray
  }

}
