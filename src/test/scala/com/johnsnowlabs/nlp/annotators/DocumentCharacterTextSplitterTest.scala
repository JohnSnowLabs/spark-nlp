package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class DocumentCharacterTextSplitterTest extends AnyFlatSpec {

  val spark = ResourceHelper.spark
  import spark.implicits._

  val text =
    "All emotions, and that\none particularly, were abhorrent to his cold, precise but\nadmirably balanced mind.\n\n" +
      "He was, I take it, the most perfect\nreasoning and observing machine that the world has seen."

  val splitTextDF = Seq(text).toDF("text")
  val documentAssembler = new DocumentAssembler().setInputCol("text")

  val textDocument: DataFrame = documentAssembler.transform(splitTextDF)

  def assertResult(text: String, result: Array[Annotation], expected: Seq[String]): Unit = {
    result.zip(expected).zipWithIndex foreach { case ((res, exChunk), i) =>
      val chunk = res.result
      assert(chunk == exChunk, "Chunk was not equal")

      val extractedChunk = text.slice(res.begin, res.end)
      assert(extractedChunk == exChunk, "Indexing is wrong")

      assert(res.metadata("document") == i.toString, "Document index not equal")
    }
  }

  val expectedDefault = Seq(
    "All emotions, and",
    "and that",
    "one particularly,",
    "were abhorrent to",
    "to his cold,",
    "precise but",
    "admirably balanced",
    "mind.",
    "He was, I take it,",
    "it, the most",
    "most perfect",
    "reasoning and",
    "and observing",
    "machine that the",
    "the world has seen.")

  behavior of "DocumentCharacterTextSplitter"

  it should "split text correctly" taggedAs FastTest in {

    val splitTextDF = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(20)
      .setChunkOverlap(5)
      .transform(textDocument)
    val result: Array[Annotation] = Annotation.collect(splitTextDF, "splits").head

    assertResult(text, result, expectedDefault)
  }

  it should "split without overlap" taggedAs FastTest in {
    val splitTextDF = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(20)
      .setChunkOverlap(0)
      .transform(textDocument)
    val result: Array[Annotation] = Annotation.collect(splitTextDF, "splits").head

    val expected = Seq(
      "All emotions, and",
      "that",
      "one particularly,",
      "were abhorrent to",
      "his cold, precise",
      "but",
      "admirably balanced",
      "mind.",
      "He was, I take it,",
      "the most perfect",
      "reasoning and",
      "observing machine",
      "that the world has",
      "seen.")

    assertResult(text, result, expected)
  }

  it should "split large texts" taggedAs FastTest in {
    val sherlockHolmes: Annotation = {
      val source = Source.fromFile("src/test/resources/spell/sherlockholmes.txt")
      val text = source.mkString
      source.close()
      new Annotation("DOCUMENT", 0, text.length, text, Map.empty)
    }

    val result: Array[Annotation] = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(20000)
      .setChunkOverlap(200)
      .annotate(Seq(sherlockHolmes))
      .toArray

    val expectedIndexes = Seq(
      (0, 19994),
      (19798, 39395),
      (39371, 59242),
      (59166, 77833),
      (77835, 97769),
      (97771, 117248),
      (117250, 137242),
      (137244, 157171),
      (157112, 177109),
      (177086, 196995),
      (196997, 216123),
      (216017, 235596),
      (235443, 255437),
      (255250, 275095),
      (275055, 294605),
      (294566, 314558),
      (314402, 333834),
      (333684, 353652),
      (353480, 373432),
      (373434, 393301),
      (393303, 413092),
      (413094, 433012),
      (432824, 452401),
      (452291, 471673),
      (471675, 491660),
      (491662, 511525),
      (511527, 530670),
      (530653, 550360),
      (550362, 570263),
      (570265, 581863))

    assert(result.map(anno => (anno.begin, anno.end)) sameElements expectedIndexes)
  }

  it should "support exploded splits" taggedAs FastTest in {
    val splitTextDF = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(20)
      .setChunkOverlap(5)
      .setExplodeSplits(true)
      .transform(textDocument)

    assert(
      splitTextDF
        .select("splits")
        .count() > 1,
      "Result was not exploded")

    val results: Array[Annotation] =
      Annotation.collect(splitTextDF, "splits").flatMap(_.toIterator)

    assertResult(text, results, expectedDefault)
  }

  it should "be able to keep separators" taggedAs FastTest in {
    val splitTextDF = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(20)
      .setChunkOverlap(5)
      .setKeepSeparators(true)
      .setTrimWhitespace(false)
      .transform(textDocument)

    val result: Array[Annotation] = Annotation.collect(splitTextDF, "splits").head

    val expectedWithSeparator = Seq(
      "All emotions, and",
      " and that",
      "\none particularly,",
      " were abhorrent to",
      " to his cold,",
      " precise but",
      "\nadmirably balanced",
      " mind.",
      "\n",
      "\nHe was, I take it,",
      " it, the most",
      " most perfect",
      "\nreasoning and",
      " and observing",
      " machine that the",
      " the world has seen.")

    assertResult(text, result, expectedWithSeparator)
  }

  it should "be able to split with regex" taggedAs FastTest in {
    val sampleText = "Hello World!"
    val sampleTextDF = documentAssembler.transform(Seq(sampleText).toDF("text"))

    val splitTextDF = new DocumentCharacterTextSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setChunkSize(5)
      .setChunkOverlap(3)
      .setSplitPatterns(Array("\\W"))
      .setPatternsAreRegex(true)
      .setKeepSeparators(true)
      .transform(sampleTextDF)

    val result: Array[Annotation] = Annotation.collect(splitTextDF, "splits").head

    val expected = Seq("Hello", " World", "!")
    assertResult(sampleText, result, expected)
  }

}
