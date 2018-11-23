package com.johnsnowlabs.nlp.annotators.spell.ocr

import scala.collection.mutable
import scala.math.min


/* computes the distance from a String to a set of predefined token classes */
trait TokenClasses {

  def levenshteinDist(s1: String, s2: String)(cost:(Char, Char) => Float): Float = {
    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i * 1.0f else if (i == 0) j * 1.0f else 0.0f }

    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + 1.0f,
        dist(j)(i - 1) + 1.0f,
        dist(j - 1)(i - 1) + cost(s2(j - 1), s1(i - 1)))

    dist(s2.length)(s1.length)
  }

  /* weighted levenshtein distance */
  def wLevenshteinDist(s1:String, s2:String) = levenshteinDist(s1, s2)(genCost)

  def loadWeights(filename: String): mutable.Map[Char, mutable.Map[Char, Float]] = {

    // store word ids
    val vocabIdxs = mutable.HashMap[Char, mutable.Map[Char, Float]]()

    scala.io.Source.fromFile(filename).getLines.foreach { case line =>
      val lineFields = line.split("\\|")
      val dist = vocabIdxs.getOrElse(lineFields(0).head, mutable.Map[Char, Float]()).updated(lineFields(1).head, lineFields(2).toFloat)
      vocabIdxs.update(lineFields(0).head, dist)
    }

    vocabIdxs
  }

  /* not implemented at this time */
  var weights = mutable.Map[Char, Map[Char, Float]]()

  private def genCost(a:Char, b:Char): Float = {
    if (weights.contains(a) && weights(a).contains(b))
      weights(a)(b)
    else if (a == b) {
      0.0f
    }
    else
      1.0f
  }

  private def minimum(i1: Float, i2: Float, i3: Float) = min(min(i1, i2), i3)

}
