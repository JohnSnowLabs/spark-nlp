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

package com.johnsnowlabs.nlp.annotators.cv

import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse

import scala.io.Source

/** `src/test/resources/dolphin/reference_<release>.json`, written by
  * `scripts/dolphin/gen_parity.py` from ByteDance's demo code running under PyTorch.
  */
case class DolphinReference(
    model: String,
    repetitionPenalty: Double,
    cases: Seq[DolphinReference.Case]) {

  def find(image: String, mode: String): DolphinReference.Case =
    cases
      .find(c => c.image == image && c.mode == mode)
      .getOrElse(throw new NoSuchElementException(s"no $mode reference for $image"))
}

object DolphinReference {

  val Fixtures = "src/test/resources/dolphin"

  /** A stage-1 box as upstream `process_coordinates` places it. */
  case class Box(label: String, padded: Seq[Int], original: Seq[Int])

  /** A page element as upstream `process_elements` returns it; figures carry no text. */
  case class Element(label: String, readingOrder: Int, bbox: Seq[Int], text: String)

  /** `output` is the stage-1 string for `layout` and `page`, and the element text otherwise. */
  case class Case(
      image: String,
      mode: String,
      output: String,
      boxes: Seq[Box],
      elements: Seq[Element])

  def load(release: String): DolphinReference = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    val source = Source.fromFile(s"$Fixtures/reference_$release.json", "UTF-8")
    val json =
      try parse(source.mkString)
      finally source.close()

    DolphinReference(
      model = (json \ "model").extract[String],
      repetitionPenalty =
        (json \ "generation" \ "repetition_penalty").extractOpt[Double].getOrElse(1.0),
      cases = (json \ "cases").children.map { c =>
        Case(
          image = (c \ "image").extract[String],
          mode = (c \ "mode").extract[String],
          output = (c \ "output").extract[String],
          boxes = (c \ "boxes").children.map { b =>
            Box(
              (b \ "label").extract[String],
              (b \ "padded").extract[List[Int]],
              (b \ "original").extract[List[Int]])
          },
          elements = (c \ "elements").children.map { e =>
            Element(
              (e \ "label").extract[String],
              (e \ "reading_order").extract[Int],
              (e \ "bbox").extract[List[Int]],
              (e \ "text").extract[String])
          })
      })
  }
}
