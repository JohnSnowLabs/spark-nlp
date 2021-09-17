/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.spell.util

import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.math.min
import scala.util.Random

object Utilities {

  private val logger = LoggerFactory.getLogger("SpellCheckersUtilities")

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

  /** Computes Levenshtein distance :
    * Metric of measuring difference between two sequences (edit distance)
    * Source: https://rosettacode.org/wiki/Levenshtein_distance
    * */

  def levenshteinDistance(s1: String, s2: String): Int = {
    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i else if (i == 0) j else 0 }

    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + 1, dist(j)(i - 1) + 1, dist(j - 1)(i - 1) + 1)

    dist(s2.length)(s1.length)
  }

  private def minimum(i1: Int, i2: Int, i3: Int) = min(min(i1, i2), i3)

  def limitDuplicates(duplicatesLimit: Int, text: String, overrideLimit: Option[Int] = None): String = {
    var duplicates = 0
    text.zipWithIndex.collect {
      case (w, i) =>
        if (i == 0) {
          w
        }
        else if (w == text(i - 1)) {
          if (duplicates < overrideLimit.getOrElse(duplicatesLimit)) {
            duplicates += 1
            w
          } else {
            ""
          }
        } else {
          duplicates = 0
          w
        }
    }.mkString("")
  }

  /** Possibilities analysis */
  def variants(targetWord: String): List[String] = {

    val splits = (0 to targetWord.length).map(i => (targetWord.take(i), targetWord.drop(i)))
    val vars = scala.collection.mutable.Set.empty[String]
    splits.toIterator.foreach{case (a,b) =>
      if (b.nonEmpty) {
        vars.add(a + b.tail)
      }
      if (b.length > 1) {
        vars.add(a + b(1) + b(0) + b.drop(2))
      }
      if (b.nonEmpty) {
        alphabet.foreach(c => vars.add(a + c + b.tail))
      }
      alphabet.foreach(c => vars.add(a + c + b))
    }
    vars.toList
  }

  /** possible variations of the word by removing duplicate letters */
  def reductions(word: String, reductionsLimit: Int): List[String] = {

    val flatWord = word.toCharArray.toIterator.zipWithIndex.collect {
      case (c, i) =>
        val n = numberOfDuplicates(word, i)
        if (n > 0) {
          (0 to n).map(r => c.toString * r).take(reductionsLimit).toIterator
        } else {
          Iterator(c.toString)
        }
    }

    val reds = cartesianProductNonRecursive(flatWord).map(_.mkString(""))
    logger.debug("parsed reductions: " + reds.size)
    reds.toList
  }

  private def numberOfDuplicates(text: String, id: Int): Int = {
    var idx = id
    val initialId = idx
    val last = text(idx)
    while (idx+1 < text.length && text(idx+1) == last) {
      idx += 1
    }
    idx - initialId
  }

  /** flattens vowel possibilities */
  def getVowelSwaps(word: String, vowelSwapLimit: Int): List[String] = {
    if (word.length > vowelSwapLimit) return List.empty[String]
    val flatWord: List[List[Char]] = word.toCharArray.collect {
      case c => if (vowels.contains(c)) {
        vowels.toList
      } else {
        List(c)
      }
    }.toList
    val vowelSwaps = cartesianProduct(flatWord).map(_.mkString(""))
    logger.debug("vowel swaps: " + vowelSwaps.size)
    vowelSwaps
  }

  private def cartesianProduct[T](xss: List[List[_]]): List[List[_]] = xss match {
    case Nil => List(Nil)
    case head :: tail => for (xh <- head; xt <- cartesianProduct(tail)) yield xh :: xt
  }

  private def cartesianProductNonRecursive[T](seqs: Iterator[Iterator[T]]): Seq[Seq[T]] = {
    seqs.foldLeft(Seq(Seq.empty[T]))((b, a) => b.flatMap(i => a.map(j => i ++ Seq(j))))
  }

  def getRandomValueFromList[A](list: List[A]): Option[A] = {
    list.lift(Random.nextInt(list.size))
  }

  def computeConfidenceValue[A](list: List[A]): Double = {
    1 / list.length.toDouble
  }

}
