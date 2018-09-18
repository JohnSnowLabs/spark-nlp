package com.johnsnowlabs.nlp.annotators.spell.ocr

import scala.math.min


/* computes the distance from a String to a set of predefined token classes */
trait TokenClasses {

  def levenshteinDist(s1: String, s2: String)(cost:(Char, Char) => Float): Float = {
    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i else if (i == 0) j else 0.0f }

    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + 1,
        dist(j)(i - 1) + 1,
        dist(j - 1)(i - 1) + cost(s2(j - 1), s1(j - 1)))

    dist(s2.length)(s1.length)
  }

  /* weighted levenshtein distance */
  def wLevenshteinDist(s1:String, s2:String) = levenshteinDist(s1, s2)(genCost)

  /* weighted levenshtein distance to the 'dates' regular language */
  def wLevenshteinDateDist(s1:String) = levenshteinDist(s1, "NN/NN/NNNN")(dateCost)

  /* weights for char ('k' -> Map('m' -> w))

     w = total / ccount

     total: number of times 'k' is mistaken by something else
     ccount: number of times 'k' is mistaken by 'm'

   */

  var weights = Map[Char, Map[Char, Float]]()

  private def genCost(a:Char, b:Char): Float = {
    if (weights.contains(a) && weights(a).contains(b))
      weights(a)(b)
    else if (a == b) {
      0.0f
    }
    else
      1.0f
  }

  val digits = (0 to 9).map(_.toString.head)

  private def dateCost(b:Char, a:Char): Float = {
    if (b == 'N')
      digits.map(n => genCost(a, n)).min
    else
      genCost(a, b)
  }

  private def minimum(i1: Float, i2: Float, i3: Float) = min(min(i1, i2), i3)

  val dateClass = (token:String) => {


  }
}
