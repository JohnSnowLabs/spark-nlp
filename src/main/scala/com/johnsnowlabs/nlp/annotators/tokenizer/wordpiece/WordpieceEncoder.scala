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

package com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPiece}
import scala.collection.mutable.ArrayBuffer


private[nlp] class WordpieceEncoder
(
  vocabulary: Map[String, Int],
  unkToken: String = "[UNK]",
  maxInputCharsPerWord: Int = 200,
  partPrefix: String = "##"
) {

  require(vocabulary.contains(unkToken), "token " + unkToken + " not found in vocabulary")

  def encode(token: IndexedToken): Array[TokenPiece] = {
    val unkId = vocabulary(unkToken)

    if (token.token.length > maxInputCharsPerWord)
      return Array(TokenPiece(unkToken, token.token, unkId, isWordStart = true, token.begin, token.end))

    val result = ArrayBuffer[TokenPiece]()

    val text = token.token
    var start = 0
    var end = text.length

    // Greedy search for next largest substring
    while (end > start && start < text.length) {
      val toFind = (if (start > 0) partPrefix else "") + text.substring(start, end)

      val found = vocabulary.get(toFind)
      if (found.nonEmpty) {
        val subToken = TokenPiece(toFind, token.token, found.get, start == 0,
          token.begin + start, token.begin + end - 1)
        result.append(subToken)
        start = end
        end = text.length
      } else {
        end = end - 1

        if (end == start) {
          // Not Found anything in vocabulary
          return Array(TokenPiece(unkToken, token.token, unkId, isWordStart = true, token.begin, token.end))
        }
      }
    }

    result.toArray
  }
}
