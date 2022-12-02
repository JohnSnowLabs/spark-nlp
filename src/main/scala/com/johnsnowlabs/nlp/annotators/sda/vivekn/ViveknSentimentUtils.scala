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

package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.util.io.ExternalResource
import com.johnsnowlabs.nlp.util.io.ResourceHelper.SourceStream

import java.io.FileNotFoundException
import scala.collection.mutable.{ListBuffer, Map => MMap}

trait ViveknSentimentUtils {

  /** Detects negations and transforms them into not_ form */
  def negateSequence(words: Array[String]): Set[String] = {
    val negations = Seq("not", "cannot", "no")
    val delims = Seq("?.,!:;")
    val result = ListBuffer.empty[String]
    var negation = false
    var prev: Option[String] = None
    var pprev: Option[String] = None
    words.foreach(word => {
      val processed = word.toLowerCase
      val negated = if (negation) "not_" + processed else processed
      result.append(negated)
      if (prev.isDefined) {
        val bigram = prev.get + " " + negated
        result.append(bigram)
        if (pprev.isDefined) {
          result.append(pprev.get + " " + bigram)
        }
        pprev = prev
      }
      prev = Some(negated)
      if (negations.contains(processed) || processed.endsWith("n't")) negation = !negation
      if (delims.exists(word.contains)) negation = false
    })
    result.toSet
  }

  def ViveknWordCount(
      er: ExternalResource,
      prune: Int,
      f: List[String] => Set[String],
      left: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0),
      right: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0))
      : (MMap[String, Long], MMap[String, Long]) = {
    val regex = er.options("tokenPattern").r
    val prefix = "not_"
    val sourceStream = SourceStream(er.path)
    sourceStream.content.foreach(c =>
      c.foreach(line => {
        val words = regex.findAllMatchIn(line).map(_.matched).toList
        f.apply(words)
          .foreach(w => {
            left(w) += 1
            right(prefix + w) += 1
          })
      }))
    sourceStream.close()
    if (left.isEmpty || right.isEmpty)
      throw new FileNotFoundException(
        "Word count dictionary for vivekn sentiment does not exist or is empty")
    if (prune > 0)
      (left.filter { case (_, v) => v > 1 }, right.filter { case (_, v) => v > 1 })
    else
      (left, right)
  }
}
