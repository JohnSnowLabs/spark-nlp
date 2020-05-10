package com.johnsnowlabs.nlp.annotators.sbd

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params, StringArrayParam}

import scala.collection.mutable.ArrayBuffer

/** See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic]] for further reference on how to use this API */
trait SentenceDetectorParams extends Params {

  /** whether to apply abbreviations at sentence detection */
  val useAbbrevations = new BooleanParam(this, "useAbbreviations", "whether to apply abbreviations at sentence detection")
  /** whether to only utilize custom bounds for sentence detection */
  val useCustomBoundsOnly = new BooleanParam(this, "useCustomBoundsOnly", "whether to only utilize custom bounds for sentence detection")
  /** whether to explode each sentence into a different row, for better parallelization. Defaults to false. */
  val explodeSentences = new BooleanParam(this, "explodeSentences", "whether to explode each sentence into a different row, for better parallelization. Defaults to false.")
  /** characters used to explicitly mark sentence bounds */
  val customBounds: StringArrayParam = new StringArrayParam(this, "customBounds", "characters used to explicitly mark sentence bounds")
  /** length at which sentences will be forcibly split. */
  val splitLength: IntParam = new IntParam(this, "splitLength", "length at which sentences will be forcibly split.")
  /** Set the minimum allowed length for each sentence */
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each sentence")
  /** Set the maximum allowed length for each sentence */
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each sentence")

  setDefault(
    useAbbrevations -> true,
    useCustomBoundsOnly -> false,
    explodeSentences -> false,
    customBounds -> Array.empty[String],
    minLength -> 0
  )

  /** Custom sentence separator text */
  def setCustomBounds(value: Array[String]): this.type = set(customBounds, value)

  /** Custom sentence separator text */
  def getCustomBounds: Array[String] = $(customBounds)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false. Needs customBounds. */
  def setUseCustomBoundsOnly(value: Boolean): this.type = set(useCustomBoundsOnly, value)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false. Needs customBounds. */
  def getUseCustomBoundsOnly: Boolean = $(useCustomBoundsOnly)

  /** Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to true. */
  def setUseAbbreviations(value: Boolean): this.type = set(useAbbrevations, value)

  /** Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to true. */
  def getUseAbbreviations: Boolean = $(useAbbrevations)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false. */
  def setExplodeSentences(value: Boolean): this.type = set(explodeSentences, value)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false. */
  def getExplodeSentences: Boolean = $(explodeSentences)

  /** Length at which sentences will be forcibly split */
  def setSplitLength(value: Int): this.type = set(splitLength, value)

  /** Length at which sentences will be forcibly split */
  def getSplitLength: Int = $(splitLength)


  /** Set the minimum allowed length for each sentence */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** Get the minimum allowed length for each sentence */
  def getMinLength(value: Int): Int = $(minLength)

  /** Set the maximum allowed length for each sentence */
  def setMaxLength(value: Int): this.type = {
    require(value >= ${
      minLength
    }, "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }

  /** Get the maximum allowed length for each sentence */
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
