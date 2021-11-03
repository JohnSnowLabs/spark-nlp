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

package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}

case class DictionaryFeatures(dict: Map[String, String])
{
 def get(tokens: Seq[String]): Seq[String] = {
    val lower = new StringBuilder()

    tokens.take(DictionaryFeatures.maxTokens).flatMap{token =>
      if (lower.nonEmpty)
        lower.append(" ")

      lower.append(token.toLowerCase)
      dict.get(lower.toString)
    }
  }
}

object DictionaryFeatures {
  val maxTokens = 5

  def apply(text2Feature: Seq[(String, String)]) = {
    val dict = text2Feature.map(p => (p._1.replace("-", " ").trim.toLowerCase, p._2)).toMap
    new DictionaryFeatures(dict)
  }

  def read(possibleEr: Option[ExternalResource]): DictionaryFeatures = {
    possibleEr.map(er => DictionaryFeatures(ResourceHelper.parseTupleText(er)))
      .getOrElse(new DictionaryFeatures(Map.empty[String, String]))
  }
}