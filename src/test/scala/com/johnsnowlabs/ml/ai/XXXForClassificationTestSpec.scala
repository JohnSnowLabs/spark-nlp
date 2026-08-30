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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.annotators.common.TokenPiece
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class XXXForClassificationTestSpec extends AnyFlatSpec {

  private def piece(token: String, isWordStart: Boolean): TokenPiece =
    TokenPiece(
      wordpiece = token,
      token = token,
      pieceId = 0,
      isWordStart = isWordStart,
      begin = 0,
      end = 0)

  "XXXForClassification.joinWordPieces" should "not insert a space around a WordPiece-split contraction" taggedAs FastTest in {
    // "Levi's" tokenized as ["Levi", "'", "s"], each its own word-start piece.
    val pieces = Seq(
      piece("Levi", isWordStart = true),
      piece("'", isWordStart = true),
      piece("s", isWordStart = true),
      piece("Stadium", isWordStart = true))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.vocab)

    assert(joined == "Levi's Stadium")
  }

  it should "not insert a space before closing punctuation" taggedAs FastTest in {
    val pieces = Seq(
      piece("It", isWordStart = true),
      piece("'", isWordStart = true),
      piece("s", isWordStart = true),
      piece("a", isWordStart = true),
      piece("test", isWordStart = true),
      piece(".", isWordStart = true))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.vocab)

    assert(joined == "It's a test.")
  }

  it should "drop non-word-start continuation pieces under the vocab strategy" taggedAs FastTest in {
    val pieces = Seq(
      piece("Den", isWordStart = true),
      piece("##ver", isWordStart = false),
      piece("Broncos", isWordStart = true))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.vocab)

    assert(joined == "Den Broncos")
  }

  it should "glue continuation pieces directly under the sentencePiece strategy" taggedAs FastTest in {
    val pieces = Seq(
      piece("Den", isWordStart = true),
      piece("ver", isWordStart = false),
      piece("Broncos", isWordStart = true))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.sentencePiece)

    assert(joined == "Denver Broncos")
  }

  "XXXForClassification.cleanUpTokenizationSpaces" should "leave text with no stray spacing unchanged" taggedAs FastTest in {
    assert(XXXForClassification.cleanUpTokenizationSpaces("Denver Broncos") == "Denver Broncos")
  }
}
