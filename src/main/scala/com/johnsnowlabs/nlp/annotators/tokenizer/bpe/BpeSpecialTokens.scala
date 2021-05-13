/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import scala.collection.immutable.HashSet

// TODO: How to do this properly?
private[nlp] class BpeSpecialTokens(modelType: String) {
  val availableModels: Array[String] = Array("roberta")

  def getSentencePadding: (String, String) =
    modelType match {
      case "roberta" => ("<s>", "</s>")
    }

  def getSpecialTokens: SpecialTokens =
    modelType match {
      case "roberta" => SpecialTokens.getRobertaSpecialTokens
    }
}

private[nlp] case class SpecialTokens(
                                       start: SpecialToken,
                                       end: SpecialToken,
                                       unk: SpecialToken,
                                       pad: SpecialToken,
                                       mask: SpecialToken,
                                       //                                       cls: SpecialToken,
                                     ) {
  val allTokens: HashSet[SpecialToken] = HashSet[SpecialToken](start, end, unk, pad, mask)

  def contains(s: String): Boolean = allTokens.contains(SpecialToken(content = s, id = 0))

  def iterator: Iterator[SpecialToken] = allTokens.iterator
}

private object SpecialTokens {
  def getRobertaSpecialTokens: SpecialTokens = SpecialTokens(
    SpecialToken(
      content = "<s>",
      id = 0,
    ),
    SpecialToken(
      content = "</s>",
      id = 2,
    ),
    SpecialToken(
      content = "<unk>",
      id = 3,
    ),
    SpecialToken(
      content = "<pad>",
      id = 1,
    ),
    SpecialToken(
      content = "<mask>",
      id = 50264,
      lstrip = true
    ),
  )
}

private[nlp] case class SpecialToken(
                         content: String,
                         id: Int,
                         singleWord: Boolean = false,
                         lstrip: Boolean = false,
                         rstrip: Boolean = false,
                       ) {
//  implicit def convertToString(s: SpecialToken): String = s.content
  override def hashCode(): Int = content.hashCode
  override def canEqual(that: Any): Boolean = that.isInstanceOf[SpecialToken]

  override def equals(obj: Any): Boolean = obj match {
    case obj: SpecialToken => obj.content == content
    case _ => false
  }

  override def toString: String = content
}
