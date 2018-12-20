package com.johnsnowlabs.nlp.annotators.spell.context

import scala.collection.mutable
import scala.math.min

trait WeightedLevenshtein {

  def levenshteinDist(s1: String, s2: String)(cost:(Char, Char) => Float): Float = {
    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i * 1.0f else if (i == 0) j * 1.0f else 0.0f }

    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + cost(s2(j - 1), 'Ɛ'),  // insertion
        dist(j)(i - 1) + cost('Ɛ', s1(i - 1)), // deletion
        dist(j - 1)(i - 1) + cost(s2(j - 1), s1(i - 1)))

    dist(s2.length)(s1.length)
  }

  /* weighted levenshtein distance */
  def wLevenshteinDist(s1:String, s2:String, weights:Map[Char, Map[Char, Float]]) = levenshteinDist(s1, s2)(genCost(weights))

  def loadWeights(filename: String): Map[Char, Map[Char, Float]] = {
    // store word ids
    val vocabIdxs = mutable.HashMap[Char, mutable.Map[Char, Float]]()

    scala.io.Source.fromFile(filename).getLines.foreach { case line =>
      val lineFields = line.split("\\|")
      val dist = vocabIdxs.getOrElse(lineFields(0).head, mutable.Map[Char, Float]()).updated(lineFields(1).head, lineFields(2).toFloat)
      vocabIdxs.update(lineFields(0).head, dist)
    }
    vocabIdxs.toMap.mapValues(_.toMap)
  }


  private def genCost(weights: Map[Char, Map[Char, Float]])(a:Char, b:Char): Float = {
    if (weights.contains(a) && weights(a).contains(b))
      weights(a)(b)
    else if (a == b) {
      0.0f
    }
    else
      1.0f
  }

  private def minimum(i1: Float, i2: Float, i3: Float) = min(min(i1, i2), i3)


  def learnDist(s1: String, s2: String): Seq[(String, String)] = {
    val acc: Seq[(String, String)] = Seq.empty
    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i * 1.0f else if (i == 0) j * 1.0f else 0.0f }

    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(
        dist(j - 1)(i) + 1.0f,
        dist(j)(i - 1) + 1.0f,
        dist(j - 1)(i - 1) + 1.0f)

    backTrack(dist, s2, s1, s2.length, s1.length, acc)
  }

  def backTrack(dist: Array[Array[Float]], s2:String, s1:String,
                j:Int, i:Int, acc:Seq[(String, String)]): Seq[(String, String)]= {

    if (s2(j-1) == s1(i-1)) {
      if (j == 1 && i == 1)
        acc
      else
        backTrack(dist, s2, s1, j - 1, i - 1, acc)
    }
    else {
      val pSteps = Map(dist(j - 1)(i) -> ("", s2(j - 1).toString, j - 1, i),
        dist(j)(i - 1) -> (s1(i - 1).toString, "", j, i - 1),
        dist(j - 1)(i - 1) -> (s1(i - 1).toString, s2(j - 1).toString, j - 1, i - 1))

      val best = pSteps.minBy(_._1)._2
      backTrack(dist, s2, s1, best._3, best._4, acc :+ (best._1, best._2))
    }
  }

}
