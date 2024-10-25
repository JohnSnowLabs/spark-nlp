package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFEmbeddingsTestSpec extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFEmbeddings"

  lazy val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val data = Seq(
    "The moons of Jupiter are " // "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones. The four"
  ).toDF("text").repartition(1)

  // nomic-embed-text-v1.5.f16.gguf
  def model(poolingType: String): AutoGGUFEmbeddings = AutoGGUFEmbeddings
    .loadSavedModel("models/nomic-embed-text-v1.5.f16.gguf", ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("embeddings")
    .setBatchSize(4)
    .setPoolingType(poolingType)

  def pipeline(embedModel: AutoGGUFEmbeddings = model("MEAN")) =
    new Pipeline().setStages(Array(documentAssembler, embedModel))

  it should "produce embeddings" in {
    val result = pipeline().fit(data).transform(data)
    val embeddings: Array[Float] = Annotation.collect(result, "embeddings").head.head.embeddings

    assert(embeddings != null, "embeddings should not be null")
    assert(
      embeddings.sum > 0.0,
      "embeddings should not be zero. Was there an error on llama.cpp side?")
  }

  it should "produce embeddings of different pooling types" in {
    def testPoolingType(poolingType: String): Unit = {
      val result = pipeline(model(poolingType)).fit(data).transform(data)
      val embeddings: Array[Float] = Annotation.collect(result, "embeddings").head.head.embeddings

      assert(embeddings != null, "embeddings should not be null")
      assert(
        embeddings.sum > 0.0,
        "embeddings should not be zero. Was there an error on llama.cpp side?")
    }

    Seq("NONE", "MEAN", "CLS", "LAST").foreach(testPoolingType)
  }
}
