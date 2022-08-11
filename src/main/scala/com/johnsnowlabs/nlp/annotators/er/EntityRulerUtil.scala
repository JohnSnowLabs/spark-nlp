/*
 * Copyright 2017-2022 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Success, Try}

object EntityRulerUtil {

  def mergeIntervals(intervals: List[List[Int]]): List[List[Int]] = {

    val mergedIntervals = ListBuffer[List[Int]]()
    var currentMergedInterval = List[Int]()
    val sortedIntervals = intervals.sortBy(interval => interval.head)

    sortedIntervals.zipWithIndex.foreach { case (interval, index) =>
      if (index == 0) {
        currentMergedInterval = interval
      } else {
        val mergedEnd = currentMergedInterval(1)
        val currentBegin = interval.head
        if (mergedEnd >= currentBegin) {
          val currentEnd = interval(1)
          val maxEnd = math.max(currentEnd, mergedEnd)
          currentMergedInterval = List(currentMergedInterval.head, maxEnd)
        } else {
          mergedIntervals.append(currentMergedInterval)
          currentMergedInterval = interval
        }
      }
    }

    mergedIntervals.append(currentMergedInterval)
    mergedIntervals.toList

  }

  def toBoolean(string: String): Boolean = {
    castStringToBoolean(string) match {
      case Success(value) => value
      case Failure(_) =>
        throw new IllegalArgumentException(
          "Column regex has a wrong format. It should be false or true")
    }
  }

  private def castStringToBoolean(string: String): Try[Boolean] = Try {
    string.toBoolean
  }

  private val symbols = """:$&(){}[]?/\\!><@=#-;,%_“.|'`"*#^+~€"""
  private val numbers = "0123456789"
  private val englishAlphabet = "abcdefghijklmnopqrstuvwxyz"
  private val spanishAlphabet = "abcdefghijklmnñopqrstuvwxyz" + "áéíóú"
  private val frenchAlphabet = "abcdefghijklmnopqrstuvwxyz" + "éàèùâêîôûëïüç"
  private val germanAlphabet = "abcdefghijklmnopqrstuvwxyz" + "äöüß"

  def loadAlphabet(path: String): String = {
    if (path.contains("/") | path.contains("\\")) {
      val externalResource = ExternalResource(path, ReadAs.TEXT, Map())
      val alphabet = ResourceHelper.parseLines(externalResource).mkString("")
      alphabet
    } else {
      path.toLowerCase() match {
        case "english" => englishAlphabet + englishAlphabet.toUpperCase + symbols + numbers
        case "spanish" => spanishAlphabet + spanishAlphabet.toUpperCase + symbols + numbers
        case "french" => frenchAlphabet + frenchAlphabet.toUpperCase + symbols + numbers
        case "german" => germanAlphabet + germanAlphabet.toUpperCase + symbols + numbers
        case _ =>
          throw new IllegalArgumentException(
            s"Alphabet $path not available." +
              s" Please load it using a path to a plain text file")
      }
    }
  }

}
