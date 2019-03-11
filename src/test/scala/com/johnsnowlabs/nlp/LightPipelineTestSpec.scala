package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.when
import org.scalatest._

import scala.language.reflectiveCalls

class LightPipelineTestSpec extends FlatSpec {
  def fixture = new {
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

    val model: PipelineModel = pipeline.fit(data)

    val textDF: Dataset[Row] = ContentProvider.parquetData.limit(1000)

    val textArray: Array[String] = ContentProvider.parquetData.limit(1000).select("text").as[String].collect
    val text = "hello world, this is some sentence"
  }

  "An LightPipeline" should "annotate for each annotator" in {
    val f = fixture
    val annotations = new LightPipeline(f.model).fullAnnotate(f.textArray)
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
    val f = fixture
    val annotations = new LightPipeline(f.model).annotate(f.textArray)
    assert(f.textArray.length == annotations.length)
  }

  it should "annotate single chunks of text with proper token amount" in {
    val f = fixture
    val annotations = new LightPipeline(f.model)
    val result = Benchmark.time("Time to annotate single text") {
      annotations.annotate(f.text)
    }
    assert(result("token").length == 7)
    assert(result("token")(4) == "is")
  }

  it should "run faster than a tranditional pipeline" in {
    val f = fixture

    val t1: Double = Benchmark.measure("Time to collect SparkML pipeline results") {
      f.model.transform(f.textDF).collect
    }

    val t2: Double = Benchmark.measure("Time to collect LightPipeline results in parallel") {
      new LightPipeline(f.model).annotate(f.textArray)
    }

    assert(t1 > t2)
  }
}
