package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

import scala.language.reflectiveCalls

class LightPipelineTestSpec extends FlatSpec {
  def fixtureWithNormalizer = new {
    import SparkAccessor.spark.implicits._

    val data: Dataset[Row] = ContentProvider.parquetData.limit(1000)
      .withColumn("sentiment_label", when($"sentiment".isNull or $"sentiment" === 0, "negative").otherwise("positive"))

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
      .setCorpusPrune(0)

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalizer,
        spellChecker,
        sentimentDetector
      ))

    lazy val model: PipelineModel = pipeline.fit(data)

    lazy val textDF: Dataset[Row] = ContentProvider.parquetData.limit(1000).repartition()

    lazy val textArray: Array[String] = textDF.select("text").as[String].collect()
    lazy val text = "hello world, this is some sentence"
  }

  def fixtureWithoutNormalizer = new {
    import SparkAccessor.spark.implicits._

    val data: Dataset[Row] = ContentProvider.parquetData.limit(1000)
      .withColumn("sentiment_label", when($"sentiment".isNull or $"sentiment" === 0, "negative").otherwise("positive"))

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
      .setCorpusPrune(0)

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        spellChecker,
        sentimentDetector
      ))

    lazy val model: PipelineModel = pipeline.fit(data)

    lazy val textDF: Dataset[Row] = ContentProvider.parquetData.limit(1000)

    lazy val textArray: Array[String] = textDF.select("text").as[String].collect
    lazy val text = "hello world, this is some sentence"
  }

  "An LightPipeline with normalizer" should "annotate for each annotator" in {
    val annotations = new LightPipeline(fixtureWithNormalizer.model).fullAnnotate(fixtureWithNormalizer.textArray)
    annotations.foreach { mapAnnotations =>
      mapAnnotations.values.foreach { annotations =>
        annotations.foreach { annotation =>
          assert(annotation.isInstanceOf[Annotation])
          assert(annotation.begin >= 0)
        }
      }
    }
  }

  it should "annotate for each string in the text array" in {
    val annotations = new LightPipeline(fixtureWithNormalizer.model).annotate(fixtureWithNormalizer.textArray)
    assert(fixtureWithNormalizer.textArray.length == annotations.length)
  }

  it should "annotate single chunks of text with proper token amount" in {
    val annotations = new LightPipeline(fixtureWithNormalizer.model)
    val result = Benchmark.time("Time to annotate single text") {
      annotations.annotate(fixtureWithNormalizer.text)
    }
    assert(result("token").length == 7)
    assert(result("token")(4) == "is")
  }

  it should "run faster than a tranditional pipelineWithNormalizer" in {
    val t1: Double = Benchmark.measure("Time to collect SparkML pipelineWithNormalizer results") {
      fixtureWithNormalizer.model.transform(fixtureWithNormalizer.textDF).collect
    }

    val t2: Double = Benchmark.measure("Time to collect LightPipeline results in parallel") {
      new LightPipeline(fixtureWithNormalizer.model).annotate(fixtureWithNormalizer.textArray)
    }

    assert(t1 > t2)
  }

  "An LightPipeline without normalizer" should "annotate for each annotator" in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model).fullAnnotate(fixtureWithoutNormalizer.textArray)
    annotations.foreach { mapAnnotations =>
      mapAnnotations.values.foreach { annotations =>
        assert(annotations.nonEmpty)
        annotations.foreach { annotation =>
          assert(annotation.isInstanceOf[Annotation])
          assert(annotation.begin >= 0)
        }
      }
    }
  }

  it should "annotate for each string in the text array" in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model).annotate(fixtureWithoutNormalizer.textArray)
    assert(fixtureWithoutNormalizer.textArray.length == annotations.length)
  }

  it should "annotate single chunks of text with proper token amount" in {
    val annotations = new LightPipeline(fixtureWithoutNormalizer.model)
    val result = Benchmark.time("Time to annotate single text") {
      annotations.annotate(fixtureWithoutNormalizer.text)
    }
    assert(result("token").length == 7)
    assert(result("token")(4) == "is")
  }

  it should "run faster than a tranditional pipelineWithoutNormalizer" in {
    val t1: Double = Benchmark.measure("Time to collect SparkML pipelineWithoutNormalizer results") {
      fixtureWithoutNormalizer.model.transform(fixtureWithoutNormalizer.textDF).collect
    }

    val t2: Double = Benchmark.measure("Time to collect LightPipeline results in parallel") {
      new LightPipeline(fixtureWithoutNormalizer.model).annotate(fixtureWithoutNormalizer.textArray)
    }

    assert(t1 > t2)
  }

}
