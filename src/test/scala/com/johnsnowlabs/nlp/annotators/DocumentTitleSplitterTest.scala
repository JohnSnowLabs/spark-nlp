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
package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.reader.{ElementType, Reader2Doc}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class DocumentTitleSplitterTest extends AnyFlatSpec with SparkSessionTest {

  private def doc(
      text: String,
      elementType: Option[String] = None,
      sentence: Int = 0,
      pageNumber: Option[Int] = Some(1),
      fileName: String = "sample.md"): Annotation = {
    Annotation(
      annotatorType = "document",
      begin = 0,
      end = text.length - 1,
      result = text,
      metadata = Map("sentence" -> sentence.toString, "fileName" -> fileName) ++
        elementType.map("elementType" -> _) ++
        pageNumber.map(page => "pageNumber" -> page.toString),
      embeddings = Array.emptyFloatArray)
  }

  behavior of "DocumentTitleSplitter"

  it should "group element-level documents by title metadata" taggedAs FastTest in {
    val annotations = Seq(
      doc("Overview", Some(ElementType.TITLE), sentence = 1),
      doc(
        "Unstructured can parse Markdown into elements. This makes it easy to experiment locally.",
        Some(ElementType.NARRATIVE_TEXT),
        sentence = 2),
      doc("Configuration", Some(ElementType.TITLE), sentence = 3),
      doc("max_characters controls hard chunk size.", Some(ElementType.NARRATIVE_TEXT), 4),
      doc(
        "new_after_n_chars controls a softer preferred limit.",
        Some(ElementType.NARRATIVE_TEXT),
        5),
      doc("Example", Some(ElementType.TITLE), sentence = 6),
      doc(
        "The chunk_by_title function uses those Title elements as section boundaries.",
        Some(ElementType.NARRATIVE_TEXT),
        7))

    val result = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .annotate(annotations)

    assert(result.length == 3)
    assert(
      result
        .map(_.result)
        .sameElements(Seq(
          "Overview Unstructured can parse Markdown into elements. This makes it easy to experiment locally.",
          "Configuration max_characters controls hard chunk size. new_after_n_chars controls a softer preferred limit.",
          "Example The chunk_by_title function uses those Title elements as section boundaries.")))
    assert(result.head.metadata("sectionTitle") == "Overview")
    assert(result(1).metadata("sectionTitle") == "Configuration")
    assert(result.head.metadata("fileName") == "sample.md")
    assert(!result.head.metadata.contains("elementType"))
  }

  it should "support a custom join string" taggedAs FastTest in {
    val annotations = Seq(
      doc("Configuration", Some(ElementType.TITLE), sentence = 1),
      doc("max_characters controls hard chunk size.", Some(ElementType.NARRATIVE_TEXT), 2))

    val result = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setJoinString(" | ")
      .annotate(annotations)

    assert(result.length == 1)
    assert(result.head.result == "Configuration | max_characters controls hard chunk size.")
  }

  it should "treat the full input as one section when title metadata is missing" taggedAs FastTest in {
    val annotations = Seq(
      doc("alpha beta gamma delta epsilon zeta", elementType = None, sentence = 1),
      doc("eta theta iota kappa lambda mu", elementType = None, sentence = 2))

    val noOverflow = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setMaxCharacters(20)
      .annotate(annotations)

    assert(noOverflow.length == 1)

    val withOverflow = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setEnableOverflowSplitting(true)
      .setMaxCharacters(20)
      .annotate(annotations)

    assert(withOverflow.length > 1)
    assert(withOverflow.forall(_.result.length <= 20))
  }

  it should "optionally split on page changes" taggedAs FastTest in {
    val annotations = Seq(
      doc("Overview", Some(ElementType.TITLE), sentence = 1, pageNumber = Some(1)),
      doc(
        "Page one narrative.",
        Some(ElementType.NARRATIVE_TEXT),
        sentence = 2,
        pageNumber = Some(1)),
      doc(
        "Page two narrative.",
        Some(ElementType.NARRATIVE_TEXT),
        sentence = 3,
        pageNumber = Some(2)))

    val defaultResult = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .annotate(annotations)

    assert(defaultResult.length == 1)

    val splitResult = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setSplitOnPageChange(true)
      .annotate(annotations)

    assert(splitResult.length == 2)
    assert(splitResult.head.result == "Overview Page one narrative.")
    assert(splitResult(1).result == "Page two narrative.")
  }

  it should "work end-to-end with Reader2Doc when docs are not exploded" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/markdown")
      .setContentPath("src/test/resources/reader/md/title-chunking.md")
      .setOutputCol("document")
      .setOutputAsDocument(false)
      .setExplodeDocs(false)

    val splitter = new DocumentTitleSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setExplodeSplits(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc, splitter))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val actual = AssertAnnotations.getActualResult(resultDf, "splits").flatMap(_.map(_.result))

    assert(actual.length == 3)
    assert(actual.head.startsWith("Overview"))
    assert(actual(1).startsWith("Configuration"))
    assert(actual(2).startsWith("Example"))
  }
}
