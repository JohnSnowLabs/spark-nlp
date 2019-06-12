package com.johnsnowlabs.nlp.annotators.spell.norvig

import org.slf4j.LoggerFactory
import scala.util.Random

object Utilities {

  private val logger = LoggerFactory.getLogger("NorvigApproachUtilities")

  private val alphabet = "abcdefghijjklmnopqrstuvwxyz".toCharArray
  private val vowels = "aeiouy".toCharArray

  /** distance measure between two words */
  def computeHammingDistance(word1: String, word2: String): Long =
    if (word1 == word2) 0
    else word1.zip(word2).count { case (c1, c2) => c1 != c2 } + (word1.length - word2.length).abs

  /** retrieve frequency */
  def getFrequency(word: String, wordCount: Map[String, Long]): Long = {
    wordCount.getOrElse(word, 0)
  }

  /** number of items duplicated in some text */
  def cartesianProduct[T](xss: List[List[_]]): List[List[_]] = xss match {
    case Nil => List(Nil)
    case h :: t => for (xh <- h; xt <- cartesianProduct(t)) yield xh :: xt
  }

  def numberOfDups(text: String, id: Int): Int = {
    var idx = id
    val initialId = idx
    val last = text(idx)
    while (idx+1 < text.length && text(idx+1) == last) {
      idx += 1
    }
    idx - initialId
  }

  def limitDups(dupsLimit: Int, text: String, overrideLimit: Option[Int] = None): String = {
    var dups = 0
    text.zipWithIndex.collect {
      case (w, i) =>
        if (i == 0) {
          w
        }
        else if (w == text(i - 1)) {
          if (dups < overrideLimit.getOrElse(dupsLimit)) {
            dups += 1
            w
          } else {
            ""
          }
        } else {
          dups = 0
          w
        }
    }.mkString("")
  }

  /** Possibilities analysis */
  def variants(targetWord: String): Set[String] = {
    val splits = (0 to targetWord.length).map(i => (targetWord.take(i), targetWord.drop(i)))
    val deletes = splits.collect {
      case (a,b) if b.length > 0 => a + b.tail
    }
    val transposes = splits.collect {
      case (a,b) if b.length > 1 => a + b(1) + b(0) + b.drop(2)
    }
    val replaces = splits.collect {
      case (a, b) if b.length > 0 => alphabet.map(c => a + c + b.tail)
    }.flatten
    val inserts = splits.collect {
      case (a, b) => alphabet.map(c => a + c + b)
    }.flatten
    val vars = Set(deletes ++ transposes ++ replaces ++ inserts :_ *)
    logger.debug("variants proposed: " + vars.size)
    vars
  }

  /** possible variations of the word by removing duplicate letters */
  /* ToDo: convert logic into an iterator, probably faster */
  def reductions(word: String, reductLimit: Int): Set[String] = {
    val flatWord: List[List[String]] = word.toCharArray.toList.zipWithIndex.collect {
      case (c, i) =>
        val n = Utilities.numberOfDups(word, i)
        if (n > 0) {
          (0 to n).map(r => c.toString*r).take(reductLimit).toList
        } else {
          List(c.toString)
        }
    }
    val reds = Utilities.cartesianProduct(flatWord).map(_.mkString("")).toSet
    logger.debug("parsed reductions: " + reds.size)
    reds
  }

  /** flattens vowel possibilities */
  def vowelSwaps(word: String, vowelSwapLimit: Int): Set[String] = {
    if (word.length > vowelSwapLimit) return Set.empty[String]
    val flatWord: List[List[Char]] = word.toCharArray.collect {
      case c => if (vowels.contains(c)) {
        vowels.toList
      } else {
        List(c)
      }
    }.toList
    val vswaps = Utilities.cartesianProduct(flatWord).map(_.mkString("")).toSet
    logger.debug("vowel swaps: " + vswaps.size)
    vswaps
  }

  def getRandomValueFromList[A](list: List[A]): Option[A] = {
    list.lift(Random.nextInt(list.size))
  }

  def computeConfidenceValue[A](list: List[A]): Double = {
    1 / list.length.toDouble
  }

}
