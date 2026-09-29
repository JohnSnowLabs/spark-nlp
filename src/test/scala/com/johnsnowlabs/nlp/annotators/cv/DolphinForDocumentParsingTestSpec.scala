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

import com.johnsnowlabs.ml.ai.util.DolphinUtils
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, ImageAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.json4s.jackson.JsonMethods.parse
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

/** End-to-end tests against the default pretrained model (Dolphin 1.5). Expected text comes from
  * `reference_1.5.json`.
  */
class DolphinForDocumentParsingTestSpec extends AnyFlatSpec {

  private val fixtures = DolphinReference.Fixtures
  private lazy val reference = DolphinReference.load("1.5")
  private val spark = ResourceHelper.spark

  private def imageDf(fileName: String): DataFrame =
    ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(s"$fixtures/$fileName")

  private def run(model: DolphinForDocumentParsing, fileName: String): Seq[Annotation] = {
    val assembler = new ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
    val pipeline = new Pipeline().setStages(Array(assembler, model))
    val result = pipeline.fit(imageDf(fileName)).transform(imageDf(fileName))
    result.select("elements").show(truncate = false)
    Annotation.collect(result, "elements").head.toSeq
  }

  /** Loaded once: the model holds ~2.6 GB of native ONNX Runtime memory that `-Xmx` does not
    * bound. Tests run sequentially, so resetting params per test is safe.
    */
  private lazy val model: DolphinForDocumentParsing =
    DolphinForDocumentParsing
      .pretrained()
      .setInputCols("image_assembler")
      .setOutputCol("elements")

  private def configured(
      parsingMode: String,
      outputFormat: String = "elements"): DolphinForDocumentParsing =
    model
      .setParsingMode(parsingMode)
      .setOutputFormat(outputFormat)
      .setElementBatchSize(4)
      .setMaxOutputLength(4096)

  // ---------------------------------------------------------------- element level

  "DolphinForDocumentParsing" should "parse a cropped table to HTML with Reader2Table json" taggedAs SlowTest in {
    val annotations = run(configured("table"), "table_1.jpeg")
    assert(annotations.length == 1)
    val table = annotations.head
    assert(table.result == reference.find("table_1.jpeg", "table").output)
    assert(table.metadata("elementType") == "Table")
    assert(table.metadata("dolphinLabel") == "tab")
    assert((parse(table.metadata("tableJson")) \ "rows").children.nonEmpty)
  }

  it should "read formula and code crops with their own prompts" taggedAs SlowTest in {
    val formula = run(configured("formula"), "line_formula.jpeg").head
    assert(formula.result == reference.find("line_formula.jpeg", "formula").output)
    assert(formula.metadata("dolphinLabel") == "equ")
    val code = run(configured("code"), "code.jpeg").head
    assert(code.result == reference.find("code.jpeg", "code").output)
    assert(code.metadata("dolphinLabel") == "code")
  }

  // ---------------------------------------------------------------- layout

  it should "detect layout elements without transcribing them" taggedAs SlowTest in {
    val annotations = run(configured("layout"), "page_1.jpeg")
    val expected = reference.find("page_1.jpeg", "layout").boxes
    assert(annotations.map(_.metadata("dolphinLabel")) == expected.map(_.label))
    assert(annotations.map(_.metadata("bbox")) == expected.map(_.original.mkString(",")))
    assert(annotations.forall(_.result.isEmpty), "layout mode must not transcribe")
  }

  // ---------------------------------------------------------------- page level

  it should "parse a full page into elements in reading order" taggedAs SlowTest in {
    val annotations = run(configured("page"), "page_1.jpeg")
    val expected = reference.find("page_1.jpeg", "page").elements
    assert(
      annotations.map(a =>
        (a.metadata("readingOrder").toInt, a.metadata("dolphinLabel"), a.metadata("bbox"))) ==
        expected.map(e => (e.readingOrder, e.label, e.bbox.mkString(","))))
    assert(annotations.map(_.result) == expected.map(_.text))
    annotations.zip(annotations.tail).foreach { case (a, b) =>
      assert(b.begin == a.begin + a.result.length + 1, "elements are laid end to end")
    }
  }

  it should "assemble a page into markdown" taggedAs SlowTest in {
    val annotations = run(configured("page", outputFormat = "markdown"), "page_1.jpeg")
    assert(annotations.length == 1, "markdown mode emits one annotation per page")
    val elements = reference.find("page_1.jpeg", "page").elements
    assert(
      annotations.head.result == DolphinUtils.toMarkdown(elements.map(e => (e.label, e.text))))
    assert(annotations.head.metadata("elementCount").toInt == elements.length)
  }

  // ---------------------------------------------------------------- contract

  it should "survive a save/load round trip" taggedAs SlowTest in {
    val before = run(configured("table"), "table_1.jpeg")

    // This is the one test that must hold a second copy of the model: saving writes ~2.6 GB and
    // loading it back builds a second set of ONNX sessions. Clean both up rather than leaving them
    // for the rest of the suite to run alongside.
    val saved = "./tmp_dolphin_model"
    try {
      configured("table").write.overwrite().save(saved)
      val reloaded = DolphinForDocumentParsing
        .load(saved)
        .setInputCols("image_assembler")
        .setOutputCol("elements")

      val after = run(reloaded, "table_1.jpeg")
      assert(before.map(_.result) == after.map(_.result))
      assert(before.head.metadata("elementType") == after.head.metadata("elementType"))
    } finally {
      FileUtils.deleteQuietly(new File(saved))
    }
  }

  it should "return one row of output per input row" taggedAs SlowTest in {
    val textModel = configured("text")
    val assembler = new ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
    // Two small crops: the contract under test is row in/row out, and the page fixture would add a
    // full two-stage parse for no extra coverage.
    val df = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(s"$fixtures/para_1.jpg", s"$fixtures/line_formula.jpeg")
    val result = new Pipeline().setStages(Array(assembler, textModel)).fit(df).transform(df)
    assert(df.count() == 2)
    assert(result.count() == df.count(), "batchAnnotate must not drop or duplicate rows")
  }

  // Elements finish at wildly different lengths. Without a per-sequence finish latch the loop runs
  // every batch to maxOutputLength, because `ids.last == eos` stops holding once a finished row is
  // padded.
  it should "terminate on the shortest element, not the longest" taggedAs SlowTest in {
    val fast = configured("text").setMaxOutputLength(512)
    val started = System.currentTimeMillis()
    val annotations = run(fast, "para_1.jpg")
    val elapsed = System.currentTimeMillis() - started
    assert(annotations.head.result.nonEmpty)
    assert(
      elapsed < 120000,
      s"a 37-character line took ${elapsed}ms; the finish latch may be broken")
  }
}
