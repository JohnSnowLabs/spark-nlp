package com.johnsnowlabs.nlp.annotators.keyword.yake

import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam, Params, StringArrayParam}

trait YakeParams extends Params {

  val windowSize = new IntParam(this, "windowSize", "Window size for Co-Occurrence")
  val maxNGrams = new IntParam(this, "maxNGrams", "Maximum N-grams a keyword should have")
  val minNGrams = new IntParam(this, "minNGrams", "Minimum N-grams a keyword should have")
  val nKeywords = new IntParam(this, "nKeywords", "Number of Keywords to extract")
  val threshold = new FloatParam(this, "threshold", "Threshold to filter keywords")
  val stopWords: StringArrayParam =
    new StringArrayParam(this, "stopWords", "the words to be filtered out. by default it's english stop words from Spark ML")
  def setStopWords(value: Array[String]): this.type = set(stopWords, value)
  def getStopWords: Array[String] = $(stopWords)

  def setWindowSize(value: Int): this.type = set(windowSize, value+1)
  def setMaxNGrams(value: Int): this.type = set(maxNGrams, value)
  def setMinNGrams(value: Int): this.type = set(minNGrams, value)
  def setNKeywords(value: Int): this.type = set(nKeywords, value)
  def setThreshold(value: Float): this.type = set(threshold,value)
}
