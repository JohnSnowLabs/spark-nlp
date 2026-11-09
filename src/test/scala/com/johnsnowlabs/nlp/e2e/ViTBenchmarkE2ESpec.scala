package com.johnsnowlabs.nlp.e2e

import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification
import com.johnsnowlabs.nlp.benchmark.{Benchmark, BenchmarkTask}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.scalatest.flatspec.AnyFlatSpec

/** Kept outside the test class: a UDF closure defined as a method value inside an `AnyFlatSpec`
  * captures the enclosing (non-serializable, ScalaTest-internal-state-holding) test instance.
  */
object ViTBenchmarkE2ESpec {
  val goldStandards: Map[String, String] = Map(
    "palace.JPEG" -> "palace",
    "egyptian_cat.jpeg" -> "Egyptian cat",
    "hippopotamus.JPEG" -> "hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "hen.JPEG" -> "hen",
    "ostrich.JPEG" -> "ostrich, Struthio camelus",
    "junco.JPEG" -> "junco, snowbird",
    "bluetick.jpg" -> "bluetick",
    "chihuahua.jpg" -> "Chihuahua",
    "tractor.JPEG" -> "tractor",
    "ox.JPEG" -> "ox")
}

/** End-to-end test combining PR 14849 (Benchmark.evaluate/ImageClassification) with PR 14851
  * (ViTClassifier's Option-stringification fix): confirms real model metadata keys are clean (not
  * "Some(...)"/"None") and that Benchmark.evaluate's rankedLabels/top-k logic scores a real
  * pretrained model's real output correctly end-to-end.
  */
class ViTBenchmarkE2ESpec extends AnyFlatSpec {

  private val goldStandards = ViTBenchmarkE2ESpec.goldStandards

  "Benchmark.evaluate" should "score a real pretrained ViT model end-to-end with clean metadata" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val imageDF = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load("src/test/resources/image/")

    // A Scala-closure UDF instead of a generated SQL map literal: several gold labels contain
    // commas (e.g. "hippopotamus, hippo, river horse, Hippopotamus amphibius"), which is exactly
    // the kind of value a hand-built `map('k', 'v', ...)` SQL string is easy to get wrong.
    val goldLookup = udf((fileName: String) => ViTBenchmarkE2ESpec.goldStandards.get(fileName))
    // Cached and materialized before being reused across both pipeline.fit and model.transform:
    // the "image" datasource's file-listing order isn't guaranteed stable across independent
    // re-executions of this lazy DataFrame, so without caching, the row that got a given gold
    // label here could silently line up with a *different* image by the time transform() re-reads
    // from disk -- corrupting every pairing without raising any error.
    val gold = imageDF
      .withColumn("goldFileName", element_at(split(col("image.origin"), "/"), -1))
      .withColumn("label", goldLookup(col("goldFileName")))
      .filter(col("label").isNotNull)
      .cache()
    gold.count()

    val imageAssembler = new ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
    val vit = ViTForImageClassification
      .pretrained()
      .setInputCols("image_assembler")
      .setOutputCol("class")

    val pipeline = new Pipeline().setStages(Array(imageAssembler, vit))
    val model = pipeline.fit(gold)
    val predicted = model.transform(gold)

    // Root-cause check (PR 14851): real metadata keys must be clean labels, never "Some(...)"/
    // "None" -- the exact defect Benchmark.unwrapLabelKey used to work around.
    val sampleMetadata = predicted
      .selectExpr("explode(class) as c")
      .selectExpr("map_keys(c.metadata) as keys")
      .as[Seq[String]]
      .collect()
      .flatten
    val reserved =
      Set("sentence", "image", "chunk", "score", "height", "width", "nChannels", "mode", "origin")
    val labelKeys = sampleMetadata.filterNot(reserved.contains)
    assert(labelKeys.nonEmpty, "expected label-score metadata keys")
    assert(
      !labelKeys.exists(k => k == "None" || (k.startsWith("Some(") && k.endsWith(")"))),
      s"found stringified-Option metadata keys, PR 14851 regressed: ${labelKeys.filter(k => k == "None" || k.startsWith("Some(")).distinct}")

    // End-to-end check (PR 14849 + fix #4): Benchmark.evaluate must score real output correctly.
    val report =
      Benchmark.evaluate(
        model,
        gold,
        BenchmarkTask.ImageClassification,
        textCol = "image",
        topK = 3)
    println(report)

    assert(report.support == goldStandards.size.toLong)
    assert(
      report.overall("accuracy") >= 0.8,
      s"expected high top-1 accuracy on a curated ImageNet fixture, got ${report.overall("accuracy")}")
    assert(report.overall("top3Accuracy") >= report.overall("accuracy"))
  }
}
