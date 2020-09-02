package com.johnsnowlabs.nlp.annotators.keyword.yake.util

import scala.collection.mutable
import scala.math.{log, max}

class Token(var token: String,
            var termFrequency: Int,
            var totalSentences: Int,
            var meanTF: Double,
            var stdTF: Double,
            var maxTF: Double,
            var leftCO: mutable.Map[String, Int],
            var rightCO: mutable.Map[String, Int]) {
  var nCount = 0
  var aCount = 0
  var medianSentenceOffset = 0
  var numberOfSentences = 0

  def TCase(): Double = {
    max(nCount, aCount).toDouble / (1 + log(termFrequency))
  }

  def TPosition(): Double = {
    log(3 + medianSentenceOffset)
  }

  def TFNorm(): Double = {
    termFrequency.toDouble / (meanTF + stdTF)
  }

  def TSentence(): Double = {
    numberOfSentences.toDouble / totalSentences.toDouble
  }

  def TRel(): Double = {
    1.0 + ((if (leftCO.isEmpty) 0.0 else (leftCO.size.toDouble / leftCO.values.sum.toDouble))
      + (if (rightCO.isEmpty) 0.0 else (leftCO.size.toDouble / rightCO.values.sum.toDouble)))* (termFrequency.toDouble / maxTF.toDouble)
  }

  def TScore(): Double = {
    TPosition() * TRel() / (TCase() + (TFNorm() / TRel()) + (TSentence() / TRel()))
  }
}
