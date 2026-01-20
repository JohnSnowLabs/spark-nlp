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

  it should "include custom idCol in LightPipeline results" taggedAs FastTest in {
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

    val dfResult = lightPipeline.transform(testDataset)
    assert(
      dfResult.columns.contains("colId"),
      "Expected custom idCol ('colId') to exist in the DataFrame output")

    val fullAnnotations = lightPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    assert(fullAnnotations.length == 2, "Expected 2 annotated documents from fullAnnotate")

    fullAnnotations.zip(Array(1, 2)).foreach { case (annotationMap, id) =>
      val idAnnots = annotationMap.get("colId")
      assert(idAnnots.isDefined, s"'colId' annotation should exist for ID $id")
      val idValue = idAnnots.get.head.asInstanceOf[Annotation].result
      assert(idValue == id.toString, s"'colId' should match the input ID ($id)")
    }

    val annotatedResults = lightPipeline.annotate(Array(1, 2), Array(text1, text2))
    assert(annotatedResults.length == 2, "Expected 2 annotated results from annotate()")

    annotatedResults.zip(Array(1, 2)).foreach { case (resultMap, id) =>
      assert(resultMap.contains("colId"), s"Expected 'colId' in annotate() result for ID $id")
      assert(resultMap("colId").contains(id.toString), s"'colId' value should match input ID $id")
    }
  }

  it should "include default doc_id in LightPipeline results when no idCol is set" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._
    val emptyDataSet: Dataset[_] = PipelineModels.dummyDataset

    val text1 = "Hello world. This is Spark NLP."
    val text2 = "This is another test document."
    val testDataset = Seq((1, text1), (2, text2)).toDF("idx", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))
    val model = pipeline.fit(emptyDataSet)

    val lightPipeline = new LightPipeline(model)

    val dfResult = lightPipeline.transform(testDataset)
    assert(
      !dfResult.columns.exists(_.equalsIgnoreCase("doc_id")),
      "transform() output should not include doc_id when no idCol is set and no IDs are provided")

    val fullAnnotations = lightPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    assert(fullAnnotations.length == 2, "Expected 2 documents from fullAnnotate")

    fullAnnotations.zip(Array(1, 2)).foreach { case (annotationMap, id) =>
      val idAnnots = annotationMap.get("doc_id")
      assert(idAnnots.isDefined, s"'doc_id' annotation should exist for ID $id")
      val idValue = idAnnots.get.head.asInstanceOf[Annotation].result
      assert(
        idValue == id.toString,
        s"'doc_id' should match the provided ID: expected $id, got $idValue")
    }

    val annotatedResults = lightPipeline.annotate(Array(1, 2), Array(text1, text2))
    assert(annotatedResults.length == 2, "Expected 2 annotated results")

    annotatedResults.zip(Array(1, 2)).foreach { case (resultMap, id) =>
      assert(
        resultMap.contains("doc_id"),
        s"annotate() output should include default 'doc_id' for ID $id")
      assert(
        resultMap("doc_id").contains(id.toString),
        s"'doc_id' value should match input ID $id")
    }
  }

  it should "include custom idCol and correctly filter out non-requested columns" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._

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

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer))
    val model = pipeline.fit(testDataset)

    val lightPipeline = new LightPipeline(model, outputCols = Array("sentence"))
    val dfResult = lightPipeline.transform(testDataset)

    assert(dfResult.columns.contains("colId"), "colId should exist in the DataFrame output")
    assert(dfResult.columns.contains("document"), "document should exist in the DataFrame output")
    assert(dfResult.columns.contains("sentence"), "sentence should exist in the DataFrame output")

    val filteredPipeline = new LightPipeline(model, outputCols = Array("sentence"))
    val filteredResult = filteredPipeline.transform(testDataset)

    val expectedCols = Seq("colId", "document", "sentence")
    val actualCols = filteredResult.columns.toSeq

    assert(
      actualCols.sorted.sameElements(expectedCols.sorted),
      s"Expected filtered columns: ${expectedCols.mkString(", ")}, got: ${actualCols.mkString(", ")}")

    val annotations = filteredPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))
    annotations.foreach { annotationMap =>
      val keys = annotationMap.keySet
      assert(
        keys == Set("colId", "document", "sentence"),
        s"Expected only 'colId', 'document', and 'sentence' in fullAnnotate output, got: $keys")
    }
  }

  it should "include id in document metadata and retain colId annotation" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._

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
    val model = pipeline.fit(testDataset)

    val lightPipeline = new LightPipeline(model)

    val annotations = lightPipeline.fullAnnotate(Array(1, 2), Array(text1, text2))

    assert(annotations.length == 2, "Expected two annotated documents")

    annotations.zip(Seq(1, 2)).foreach { case (annotationMap, expectedId) =>
      assert(
        annotationMap.contains("colId"),
        s"'colId' field missing in LightPipeline result for ID=$expectedId")

      val colIdAnnotation = annotationMap("colId").head
        .asInstanceOf[Annotation]

      assert(
        colIdAnnotation.result == expectedId.toString,
        s"Expected colId result=${expectedId}, got ${colIdAnnotation.result}")

      val documentAnnots = annotationMap("document")
        .map(_.asInstanceOf[Annotation])

      assert(documentAnnots.nonEmpty, "Document annotations should not be empty")

      documentAnnots.foreach { doc =>
        assert(
          doc.metadata.contains("id"),
          s"Document annotation missing 'id' metadata for ID=$expectedId")
        val metaId = doc.metadata("id")
        assert(
          metaId == expectedId.toString,
          s"Metadata id mismatch: expected $expectedId, got $metaId")
      }

      val sentenceAnnots = annotationMap("sentence")
        .map(_.asInstanceOf[Annotation])

      assert(sentenceAnnots.nonEmpty, "Sentence annotations should not be empty")

      sentenceAnnots.foreach { sent =>
        assert(
          sent.metadata.contains("id"),
          s"Sentence annotation missing 'id' metadata for ID=$expectedId")
        assert(
          sent.metadata("id") == expectedId.toString,
          s"Sentence id metadata mismatch for ID=$expectedId")
      }
    }
  }

}
