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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import scala.language.reflectiveCalls

class LightPipelineTestSpec extends AnyFlatSpec {
  def fixtureWithNormalizer = new {
    import SparkAccessor.spark.implicits._

    val data: Dataset[Row] = ContentProvider.parquetData
      .limit(1000)
      .withColumn(
        "sentiment_label",
        when($"sentiment".isNull or $"sentiment" === 0, "negative").otherwise("positive"))

    val documentAssembler: DocumentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector: SentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer: Normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker: NorvigSweetingApproach = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val sentimentDetector: ViveknSentimentApproach = new ViveknSentimentApproach()
      .setInputCols(Array("spell", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setPruneCorpus(0)

    val pipeline: Pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          normalizer,
          spellChecker,
          sentimentDetector))

    lazy val model: PipelineModel = pipeline.fit(data)

    lazy val textDF: Dataset[Row] = ContentProvider.parquetData.limit(1000).repartition()

    lazy val textArray: Array[String] = textDF.select("text").as[String].collect()
    lazy val text = "hello world, this is some sentence"
  }

  def fixtureWithoutNormalizer = new {
    import SparkAccessor.spark.implicits._

    val data: Dataset[Row] = ContentProvider.parquetData
      .limit(1000)
      .withColumn(
        "sentiment_label",
        when($"sentiment".isNull or $"sentiment" === 0, "negative").otherwise("positive"))

    val documentAssembler: DocumentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector: SentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val spellChecker: NorvigSweetingApproach = new NorvigSweetingApproach()
      .setInputCols(Array("token"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val sentimentDetector: ViveknSentimentApproach = new ViveknSentimentApproach()
      .setInputCols(Array("spell", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setPruneCorpus(0)

    val pipeline: Pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentenceDetector, tokenizer, spellChecker, sentimentDetector))

    lazy val model: PipelineModel = pipeline.fit(data)

    lazy val textDF: Dataset[Row] = ContentProvider.parquetData.limit(1000)

    lazy val textArray: Array[String] = textDF.select("text").as[String].collect
    lazy val text = "hello world, this is some sentence"
  }

  "An LightPipeline with normalizer" should "annotate for each annotator" taggedAs FastTest in {
    val annotations =
      new LightPipeline(fixtureWithNormalizer.model).fullAnnotate(fixtureWithNormalizer.textArray)
    annotations.foreach { mapAnnotations =>
      mapAnnotations.values.foreach { annotations =>
        annotations.foreach { annotation =>
          assert(annotation.isInstanceOf[Annotation])
          assert(annotation.asInstanceOf[Annotation].begin >= 0)
        }
      }
    }
    // Check that DocumentAssembler output contains sentences info
    annotations.foreach { mapAnnotations =>
      mapAnnotations.get("document").foreach { annotations =>
        annotations.map(_.asInstanceOf[Annotation]).foreach { annotation =>
          assert(annotation.metadata("sentence").toInt == 0)
        }
      }
    }
  }

  it should "annotate for each string in the text array" taggedAs FastTest in {
    val annotations =
      new LightPipeline(fixtureWithNormalizer.model).annotate(fixtureWithNormalizer.textArray)
    assert(fixtureWithNormalizer.textArray.length == annotations.length)
  }

  it should "annotate single chunks of text with proper token amount" taggedAs FastTest in {
    val annotations = new LightPipeline(fixtureWithNormalizer.model)
    val result = Benchmark.time("Time to annotate single text") {
      annotations.annotate(fixtureWithNormalizer.text)
    }
    assert(result("token").length == 7)
    assert(result("token")(4) == "is")
  }

  it should "run faster than a traditional pipelineWithNormalizer" taggedAs SlowTest in {
    val t1: Double = Benchmark.measure("Time to collect SparkML pipelineWithNormalizer results") {
      fixtureWithNormalizer.model.transform(fixtureWithNormalizer.textDF).collect
    }

    val t2: Double = Benchmark.measure("Time to collect LightPipeline results in parallel") {
      new LightPipeline(fixtureWithNormalizer.model).annotate(fixtureWithNormalizer.textArray)
    }

    assert(t1 > t2)
  }

  "An LightPipeline without normalizer" should "annotate for each annotator" taggedAs FastTest in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model)
      .fullAnnotate(fixtureWithoutNormalizer.textArray)
    annotations.foreach { mapAnnotations =>
      mapAnnotations.values.foreach { annotations =>
        assert(annotations.nonEmpty)
        annotations.foreach { annotation =>
          assert(annotation.isInstanceOf[Annotation])
          assert(annotation.asInstanceOf[Annotation].begin >= 0)
        }
      }
    }
  }

  it should "annotate for each string in the text array" taggedAs FastTest in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model)
      .annotate(fixtureWithoutNormalizer.textArray)
    assert(fixtureWithoutNormalizer.textArray.length == annotations.length)
  }

  it should "annotate single chunks of text with proper token amount" taggedAs FastTest in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model)
    val result = Benchmark.time("Time to annotate single text") {
      annotations.annotate(fixtureWithoutNormalizer.text)
    }
    assert(result("token").length == 7)
    assert(result("token")(4) == "is")
  }

  it should "run faster than a traditional pipelineWithoutNormalizer" taggedAs SlowTest in {
    val t1: Double =
      Benchmark.measure("Time to collect SparkML pipelineWithoutNormalizer results") {
        fixtureWithoutNormalizer.model.transform(fixtureWithoutNormalizer.textDF).collect
      }

    val t2: Double = Benchmark.measure("Time to collect LightPipeline results in parallel") {
      new LightPipeline(fixtureWithoutNormalizer.model)
        .annotate(fixtureWithoutNormalizer.textArray)
    }

    assert(t1 > t2)
  }

  it should "raise an error when using a wrong input size" taggedAs FastTest in {
    val lightPipeline = new LightPipeline(fixtureWithNormalizer.model)

    assertThrows[UnsupportedOperationException] {
      lightPipeline.fullAnnotate(Array("1", "2", "3"), Array("1", "2"))
    }
  }

  it should "output embeddings for LightPipeline" taggedAs SlowTest in {
    val pipeline = new PretrainedPipeline("onto_recognize_entities_bert_tiny", "en")
    val lightPipeline = new LightPipeline(pipeline.model, parseEmbeddings = true)

    val result = lightPipeline.annotate("Hello from John Snow Labs ! ")

    assert(result("embeddings").nonEmpty)

    val fullResult = lightPipeline.fullAnnotate("Hello from John Snow Labs ! ")
    val embeddingsAnnotation = fullResult("embeddings").map(_.asInstanceOf[Annotation]).head

    assert(embeddingsAnnotation.embeddings.nonEmpty)
  }

  it should "include colId in LightPipeline results and respect outputCols filtering" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._
    val emptyDataSet: Dataset[_] = PipelineModels.dummyDataset

    val text1 = "This is the first document. This is a second sentence within the first document."
    val text2 = "This is the second document."
    val testDataset = Seq((1, text1), (2, text2)).toDF("colId", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setIdCol("colId")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))
    val model = pipeline.fit(emptyDataSet)

    val lightPipeline = new LightPipeline(model)

    // Test DataFrame-based transform()
    val dfResult = lightPipeline.transform(testDataset)
    assert(
      dfResult.columns.contains("colId"),
      "colId column should exist in the DataFrame output")
    assert(
      dfResult.columns.contains("sentence"),
      "sentence column should exist in the DataFrame output")

    // Test fullAnnotate() with colId propagation
    val fullAnnotations = lightPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    assert(fullAnnotations.length == 2)

    fullAnnotations.zip(Array(1, 2)).foreach { case (annotationMap, id) =>
      val colIdAnnots = annotationMap.get("colId")
      assert(colIdAnnots.isDefined, s"colId annotation should exist for ID $id")
      val colIdValue = colIdAnnots.get.head.asInstanceOf[Annotation].result
      assert(
        colIdValue == id.toString,
        s"colId should match input ID: expected $id, got $colIdValue")
    }

    // Test annotate() with colId propagation
    val annotatedResults = lightPipeline.annotate(Array(1, 2), Array(text1, text2))
    assert(annotatedResults.length == 2)
    annotatedResults.zip(Array(1, 2)).foreach { case (resultMap, id) =>
      assert(resultMap.contains("colId"), s"annotate() result should include colId for ID $id")
      assert(resultMap("colId").contains(id.toString), s"colId should match input ID $id")
    }

    // Test LightPipeline with outputCols filtering
    val filteredPipeline = new LightPipeline(model, outputCols = Array("sentence"))
    val filteredResult = filteredPipeline.transform(testDataset)
    val expectedCols = Seq("colId", "text", "sentence")
    assert(
      filteredResult.columns.sorted.sameElements(expectedCols.sorted),
      s"Filtered DataFrame should only contain ${expectedCols.mkString(", ")}")

    val filteredAnnotations = filteredPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    filteredAnnotations.foreach { annotationMap =>
      assert(
        annotationMap.keySet == Set("sentence"),
        s"Only 'sentence' should appear in fullAnnotate output when outputCols=['sentence']")
    }
  }

  it should "filter columns when outputCols is defined" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._
    val emptyDataSet: Dataset[_] = PipelineModels.dummyDataset

    val text1 = "This is the first document. This is a second sentence within the first document."
    val text2 = "This is the second document."
    val testDataset = Seq((1, text1), (2, text2)).toDF("colId", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setIdCol("colId")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(true)

    val basePipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))
    val model = basePipeline.fit(emptyDataSet)

    val lightPipeline = new LightPipeline(model, outputCols = Array("sentence"))

    // Test DataFrame transform filtering
    val dfResult = lightPipeline.transform(testDataset)
    val expectedCols = Set("colId", "text", "sentence")
    val actualCols = dfResult.columns.toSet
    assert(
      actualCols == expectedCols,
      s"Expected DataFrame columns: ${expectedCols.mkString(", ")}, but got: ${actualCols.mkString(", ")}")

    // Test fullAnnotate filtering
    val fullAnnotations = lightPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    assert(fullAnnotations.length == 2, "Expected 2 documents in fullAnnotate output")

    fullAnnotations.foreach { annotationMap =>
      val keys = annotationMap.keySet
      assert(
        keys == Set("sentence"),
        s"Only 'sentence' should be in fullAnnotate output, found: $keys")
      val sentences = annotationMap("sentence").map(_.asInstanceOf[Annotation].result)
      assert(sentences.nonEmpty, "Sentence results should not be empty")
      assert(
        sentences.exists(_.contains("document")),
        "Sentence results should include input text content")
    }

    // Test annotate() filtering
    val annotatedResults = lightPipeline.annotate(Array(1, 2), Array(text1, text2))
    assert(annotatedResults.length == 2, "Expected 2 annotated outputs")

    annotatedResults.foreach { resultMap =>
      val keys = resultMap.keySet
      assert(
        keys == Set("sentence"),
        s"Only 'sentence' should be returned in annotate() output, found: $keys")
      val sentences = resultMap("sentence")
      assert(sentences.nonEmpty, "Sentence list should not be empty in annotate() output")
    }

  }

}
