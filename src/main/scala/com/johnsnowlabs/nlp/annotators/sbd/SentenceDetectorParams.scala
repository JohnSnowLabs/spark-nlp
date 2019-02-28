package com.johnsnowlabs.nlp.annotators.sbd

import org.apache.spark.ml.param.{BooleanParam, IntParam, Params, StringArrayParam}

trait SentenceDetectorParams extends Params {

  val useAbbrevations = new BooleanParam(this, "useAbbreviations", "whether to apply abbreviations at sentence detection")

  val useCustomBoundsOnly = new BooleanParam(this, "useCustomBoundsOnly", "whether to only utilize custom bounds for sentence detection")

  val explodeSentences = new BooleanParam(this, "explodeSentences", "whether to explode each sentence into a different row, for better parallelization. Defaults to false.")

  val customBounds: StringArrayParam = new StringArrayParam(
    this,
    "customBounds",
    "characters used to explicitly mark sentence bounds"
  )

  val maxLength: IntParam = new IntParam(this, "maxLength", "length at which sentences will be forcibly split. Defaults to 240")

  setDefault(
    useAbbrevations -> true,
    useCustomBoundsOnly -> false,
    explodeSentences -> false,
    maxLength -> 240,
    customBounds -> Array.empty[String]
  )

  def setCustomBounds(value: Array[String]): this.type = set(customBounds, value)

  def setUseCustomBoundsOnly(value: Boolean): this.type = set(useCustomBoundsOnly, value)

  def setUseAbbreviations(value: Boolean): this.type = set(useAbbrevations, value)

  def setExplodeSentences(value: Boolean): this.type = set(explodeSentences, value)

  def setMaxLength(value: Int): this.type = set(maxLength, value)

  def getMaxLength: Int = $(maxLength)

}
