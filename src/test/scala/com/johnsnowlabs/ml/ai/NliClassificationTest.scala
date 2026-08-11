/*
 * Copyright 2017-2026 John Snow Labs
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

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

/** Covers the sentence-pair encoding `SampleEntailmentMatrix` feeds its NLI model, against a
  * small hand-written vocabulary and with no ONNX model involved.
  */
class NliClassificationTest extends AnyFlatSpec {

  private val words = Seq("paris", "is", "the", "capital", "of", "france", "london", "a", "b")

  private val vocabulary: Map[String, Int] =
    (Seq("[PAD]", "[UNK]", "[CLS]", "[SEP]") ++ words).zipWithIndex.toMap

  private val cls = vocabulary("[CLS]")
  private val sep = vocabulary("[SEP]")

  private def encode(
      premise: String,
      hypothesis: String,
      maxSeqLength: Int = 512): (Array[Int], Array[Int]) =
    NliClassification.encodePair(
      vocabulary,
      caseSensitive = false,
      premise,
      hypothesis,
      maxSeqLength)

  behavior of "NliClassification.encodePair"

  it should "wrap the pair as [CLS] premise [SEP] hypothesis [SEP]" taggedAs FastTest in {
    val (ids, _) = encode("paris", "london")
    assert(ids sameElements Array(cls, vocabulary("paris"), sep, vocabulary("london"), sep))
  }

  it should "mark the premise segment 0 and the hypothesis segment 1" taggedAs FastTest in {
    val (ids, typeIds) = encode("paris is", "london")
    assert(ids.length == typeIds.length)
    // [CLS] paris is [SEP] -> 0, london [SEP] -> 1
    assert(typeIds sameElements Array(0, 0, 0, 0, 1, 1))
  }

  it should "lowercase when caseSensitive is false" taggedAs FastTest in {
    val (lower, _) = encode("paris", "london")
    val (upper, _) =
      NliClassification.encodePair(vocabulary, caseSensitive = false, "PARIS", "LONDON", 512)
    assert(lower sameElements upper)
  }

  it should "produce ids and segment ids of equal length for every input" taggedAs FastTest in {
    val cases = Seq(("", ""), ("paris", ""), ("", "paris"), ("paris is the capital", "london"))
    cases.foreach { case (premise, hypothesis) =>
      val (ids, typeIds) = encode(premise, hypothesis)
      assert(ids.length == typeIds.length, s"mismatch for ('$premise', '$hypothesis')")
    }
  }

  it should "handle empty strings without throwing" taggedAs FastTest in {
    val (ids, typeIds) = encode("", "")
    assert(ids sameElements Array(cls, sep, sep))
    assert(typeIds sameElements Array(0, 0, 1))
  }

  it should "never exceed maxSeqLength" taggedAs FastTest in {
    val long = Seq.fill(200)("paris").mkString(" ")
    Seq(8, 16, 64, 512).foreach { limit =>
      val (ids, typeIds) = encode(long, long, limit)
      assert(ids.length <= limit, s"ids exceeded maxSeqLength $limit")
      assert(typeIds.length <= limit)
    }
  }

  it should "cap the premise at half the budget so a long premise cannot crowd out the hypothesis" taggedAs FastTest in {
    // budget = 16 - 3 = 13, so the premise takes at most 6 pieces and the hypothesis gets 7.
    val longPremise = Seq.fill(50)("paris").mkString(" ")
    val hypothesis = Seq.fill(50)("london").mkString(" ")
    val (ids, typeIds) = encode(longPremise, hypothesis, 16)
    assert(ids.length == 16)
    assert(typeIds.count(_ == 0) == 8) // [CLS] + 6 premise pieces + first [SEP]
    assert(typeIds.count(_ == 1) == 8) // 7 hypothesis pieces + trailing [SEP]
  }

  it should "give the hypothesis the whole remaining budget when the premise is short" taggedAs FastTest in {
    val hypothesis = Seq.fill(50)("london").mkString(" ")
    val (ids, typeIds) = encode("paris", hypothesis, 16)
    assert(ids.length == 16)
    assert(typeIds.count(_ == 0) == 3) // [CLS] + 1 premise piece + first [SEP]
  }

  it should "always keep the three special tokens even at the smallest usable length" taggedAs FastTest in {
    val (ids, _) = encode("paris is the capital", "london is too", 3)
    assert(ids sameElements Array(cls, sep, sep))
  }

  it should "map out-of-vocabulary words to [UNK] rather than dropping them" taggedAs FastTest in {
    val (ids, _) = encode("zzzz", "paris")
    assert(ids.contains(vocabulary("[UNK]")))
    assert(ids.length == 5)
  }
}
