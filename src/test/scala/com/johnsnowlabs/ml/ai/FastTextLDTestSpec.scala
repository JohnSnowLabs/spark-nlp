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
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class FastTextLDTestSpec extends AnyFlatSpec {

  private implicit val formats: DefaultFormats.type = DefaultFormats

  "FastTextFeaturizer.hash" should "be fastText's FNV-1a over signed bytes" taggedAs FastTest in {
    assert(FastTextFeaturizer.hash("".getBytes("UTF-8")) == 2166136261L)
    assert(FastTextFeaturizer.hash("<a>".getBytes("UTF-8")) == 2359087600L)
    // unsigned bytes would give 4222937559
    assert(FastTextFeaturizer.hash("<é>".getBytes("UTF-8")) == 1002806231L)
  }

  "FastTextFeaturizer" should "give fastText's ids" taggedAs FastTest in {
    // expected ids from fastText's get_line and get_subwords with GlotLID v4
    val source =
      Source.fromFile(
        "src/test/resources/language-detector/glotlid/featurizer_cases.json",
        "UTF-8")
    val json =
      try parse(source.mkString)
      finally source.close()
    val featurizer = new FastTextFeaturizer(
      vocabulary = (json \ "vocab").extract[Map[String, Int]],
      nwords = (json \ "nwords").extract[Int],
      minn = (json \ "minn").extract[Int],
      maxn = (json \ "maxn").extract[Int],
      bucket = (json \ "bucket").extract[Int])

    val cases = (json \ "cases").children
    val failures = cases.filter { c =>
      featurizer.featurize((c \ "text").extract[String]).toSeq != (c \ "ids").extract[Seq[Int]]
    }
    assert(cases.size == 433)
    assert(failures.isEmpty, failures.map(c => (c \ "name").extract[String]).mkString(", "))
  }

  private val chinese = Array("eng_Latn", "cmn_Hani", "yue_Hani", "spa_Latn")

  "FastTextLD.decide" should "pool the scores of a non-Latin script" taggedAs FastTest in {
    val probs = Array(0.10f, 0.45f, 0.40f, 0.05f)
    val text = "已向家属告知诊断"
    assert(FastTextLD.decide(probs, chinese, text, 0.7f, "unk", false, 0) == ("unk", 0.45f))
    val (label, confidence) = FastTextLD.decide(probs, chinese, text, 0.7f, "unk", true, 0)
    assert(label == "cmn_Hani")
    assert(math.abs(confidence - 0.85f) < 1e-6f)
  }

  it should "not pool Latin-script labels" taggedAs FastTest in {
    val labels = Array("eng_Latn", "spa_Latn", "por_Latn")
    val probs = Array(0.20f, 0.41f, 0.39f)
    val text = "Paciente com dor no peito ha tres dias"
    assert(FastTextLD.decide(probs, labels, text, 0.7f, "unk", true, 0) == ("unk", 0.41f))
  }

  it should "leave zxx_ labels out of the pooled score" taggedAs FastTest in {
    val labels = Array("arb_Arab", "zxx_Arab", "pes_Arab", "eng_Latn")
    val probs = Array(0.50f, 0.30f, 0.10f, 0.10f)
    val (label, confidence) =
      FastTextLD.decide(probs, labels, "المريض رجل يبلغ من العمر", 0.7f, "unk", true, 0)
    assert(label == "unk")
    assert(math.abs(confidence - 0.60f) < 1e-6f)
  }

  it should "apply minLetters to a Latin-script top label" taggedAs FastTest in {
    val labels = Array("nld_Latn", "eng_Latn")
    val probs = Array(0.84f, 0.16f)
    def decide(text: String, minLetters: Int) =
      FastTextLD.decide(probs, labels, text, 0.7f, "unk", true, minLetters)

    assert(decide("No oral lesions.", 20) == ("unk", 0.84f))
    assert(decide("No oral lesions.", 0) == ("nld_Latn", 0.84f))
    assert(decide("abcdefghij abcdefghi", 20) == ("unk", 0.84f))
    assert(decide("abcdefghij abcdefghij", 20) == ("nld_Latn", 0.84f))
    assert(decide("β-HCG negative.", 20) == ("unk", 0.84f))
    assert(decide("B12 500 µg daily.", 20) == ("unk", 0.84f))
  }

  it should "not apply minLetters to other top labels" taggedAs FastTest in {
    def decide(label: String, text: String) =
      FastTextLD.decide(
        Array(0.90f, 0.10f),
        Array(label, "eng_Latn"),
        text,
        0.7f,
        "unk",
        true,
        20)

    assert(decide("cmn_Hani", "已向家属") == ("cmn_Hani", 0.90f))
    assert(decide("cmn_Hani", "Pt 张伟") == ("cmn_Hani", 0.90f))
    assert(decide("ell_Grek", "Ναι.") == ("ell_Grek", 0.90f))
    assert(decide("zxx_Zsym", "+++ ###") == ("zxx_Zsym", 0.90f))
  }

  it should "use thresholdLabel below the threshold" taggedAs FastTest in {
    val labels = Array("fra_Latn", "eng_Latn")
    val text = "Le patient est un homme de soixante ans"
    assert(
      FastTextLD.decide(Array(0.75f, 0.25f), labels, text, 0.75f, "xx", false, 0) ==
        ("fra_Latn", 0.75f))
    assert(
      FastTextLD.decide(Array(0.74f, 0.26f), labels, text, 0.75f, "xx", false, 0) ==
        ("xx", 0.74f))
  }

  "FastTextLD.topScores" should "return the k best labels, or all of them for k = 0" taggedAs FastTest in {
    val labels = Array("a", "b", "c", "d")
    val probs = Array(0.1f, 0.4f, 0.2f, 0.3f)
    assert(FastTextLD.topScores(probs, labels, 2) == Seq(("b", 0.4f), ("d", 0.3f)))
    assert(FastTextLD.topScores(probs, labels, 0).map(_._1) == Seq("b", "d", "c", "a"))
    assert(FastTextLD.topScores(probs, labels, 10).size == 4)
  }

  "FastTextLD.fitsBatch" should "not overflow" taggedAs FastTest in {
    assert(FastTextLD.fitsBatch(rows = 4, width = 16384, maxCells = 65536))
    assert(!FastTextLD.fitsBatch(rows = 5, width = 16384, maxCells = 65536))
    assert(!FastTextLD.fitsBatch(rows = 13001, width = 165000, maxCells = 65536))
  }
}
