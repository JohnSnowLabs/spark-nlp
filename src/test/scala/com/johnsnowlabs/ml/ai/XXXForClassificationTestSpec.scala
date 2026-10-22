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

import com.johnsnowlabs.nlp.annotators.common.TokenPieceTestUtils.{continuationPiece, piece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class XXXForClassificationTestSpec extends AnyFlatSpec {

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

  it should "reconstruct a word split into WordPiece continuation pieces under the vocab strategy" taggedAs FastTest in {
    // "Denver" split into "Den" + "##ver": WordpieceEncoder.encode sets `.token` to the whole
    // word ("Denver") on BOTH pieces (see WordpieceEncoder.scala), so reading it off just the
    // word-start piece already reconstructs the full word -- nothing is lost by ignoring the
    // continuation piece here.
    val pieces = Seq(
      piece("Denver", isWordStart = true),
      continuationPiece(wordpiece = "##ver", wholeWordToken = "Denver"),
      piece("Broncos", isWordStart = true))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.vocab)

    assert(joined == "Denver Broncos")
  }

  it should "not drop a span that starts mid-word on a continuation piece under the vocab strategy" taggedAs FastTest in {
    // A predicted QA/NER answer span can start on a continuation piece with no preceding
    // word-start piece in range (e.g. the model's start index landed on "##ver" of "Denver"
    // rather than "Den"). Filtering by `isWordStart` alone would silently drop it.
    val pieces = Seq(continuationPiece(wordpiece = "##ver", wholeWordToken = "Denver"))

    val joined = XXXForClassification.joinWordPieces(pieces, MergeTokenStrategy.vocab)

    assert(joined == "ver")
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

  it should "remove the stray space before a bare 't contraction piece" taggedAs FastTest in {
    // Found live against a real RoBERTa QA model: some BPE vocabularies split "Don't" as
    // "Don" + "'t" (a bare word-start "'t" piece) rather than "Do" + "n't", which HF's own
    // clean_up_tokenization list (" n't" -> "n't") doesn't cover.
    assert(XXXForClassification.cleanUpTokenizationSpaces("Don 't be evil") == "Don't be evil")
  }

  "XXXForClassification.answerSpanBounds" should "default to (0, 0) instead of throwing on an empty decodedAnswer" taggedAs FastTest in {
    // Regression test: decodedAnswer is empty when the model predicts start >= end (e.g. a
    // squad2-style "no answer" span pointing back at/near the CLS token); .head/.last used to
    // throw NoSuchElementException on that Seq.
    assert(XXXForClassification.answerSpanBounds(Seq.empty) == (0, 0))
  }

  it should "return the first piece's begin and the last piece's end for a non-empty decodedAnswer" taggedAs FastTest in {
    val pieces = Seq(piece("Denver", isWordStart = true), piece("Broncos", isWordStart = true))
      .zip(Seq((10, 16), (17, 24)))
      .map { case (p, (begin, end)) => p.copy(begin = begin, end = end) }

    assert(XXXForClassification.answerSpanBounds(pieces) == (10, 24))
  }
}
