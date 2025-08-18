package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFEmbeddingsTestSpec extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFEmbeddings"

  lazy val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val data = Seq(
    "The moons of Jupiter are ", // "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones. The four"
    "Earth is ", // "Earth is 4.5 billion years old. It has been home to countless species, some of which have gone extinct, while others have evolved into"
    "The moon is ", // "The moon is 1/400th the size of the sun. The sun is 1.39 million kilometers in diameter, while"
    "The sun is " //
  ).toDF("text").repartition(1)

  lazy val longDataCopies = 16
  lazy val longData = {
    val text = "All work and no play makes Jack a dull boy" * 100
    Seq.fill(longDataCopies)(text).toDF("text").repartition(4)
  }

  println(ResourceHelper.spark.version)
  // nomic-embed-text-v1.5.Q8_0.gguf
  def model(poolingType: String): AutoGGUFEmbeddings = AutoGGUFEmbeddings
    .loadSavedModel("models/Qwen3-Embedding-0.6B-Q8_0.gguf", ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("embeddings")
    .setBatchSize(4)
    .setPoolingType(poolingType)
    .setNCtx(8192)

  def pipeline(embedModel: AutoGGUFEmbeddings = model("MEAN")): Pipeline =
    new Pipeline().setStages(Array(documentAssembler, embedModel))

  it should "produce embeddings" taggedAs SlowTest in {
    val result = pipeline().fit(data).transform(data)
    val collected = Annotation.collect(result, "embeddings")

    collected.foreach { annotations =>
      val embeddings = annotations.head.embeddings
      assert(embeddings != null, "embeddings should not be null")
      assert(
        embeddings.sum > 0.0,
        "embeddings should not be zero. Was there an error on llama.cpp side?")
    }
  }

  it should "produce embeddings of different pooling types" taggedAs SlowTest in {
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

  it should "be serializable" taggedAs SlowTest in {

    val data = Seq("Hello, I am a").toDF("text")
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model("MEAN")))

    val pipelineModel = pipeline.fit(data)
    val savePath = "./tmp_autogguf_embedding_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFEmbeddings]
      .write
      .overwrite()
      .save(savePath)

    val loadedModel = AutoGGUFEmbeddings.load(savePath)
    val newPipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, loadedModel))

    newPipeline
      .fit(data)
      .transform(data)
      .select("embeddings.embeddings")
      .show(truncate = false)
  }

  it should "return error messages when embeddings can't be created" taggedAs SlowTest in {
    val result = pipeline().fit(longData).transform(longData)
    val collected = Annotation.collect(result, "embeddings")
    assert(collected.length == longDataCopies)

    collected.foreach { annotations =>
      assert(
        annotations.head.metadata.contains("llamacpp_exception"),
        "llamacpp_exception should be present")
    }

  }

  it should "embed long text" taggedAs SlowTest in {
    val result = pipeline(
      model("MEAN")
        .setNUbatch(4096)
        .setNBatch(4096)).fit(longData).transform(longData)
    val collected = Annotation.collect(result, "embeddings")
    assert(collected.length == longDataCopies, "Should return the same number of rows")

    collected.foreach { annotations =>
      val embeddings = annotations.head.embeddings
      assert(embeddings != null, "embeddings should not be null")
      assert(
        embeddings.sum > 0.0,
        "embeddings should not be zero. Was there an error on llama.cpp side?")
    }
  }

  it should "accept protocol prepended paths" taggedAs SlowTest in {
    val data = Seq("Hello, I am a").toDF("text")
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model("MEAN")))
    val pipelineModel = pipeline.fit(data)

    val savePath = "file:///tmp/tmp_autogguf_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFEmbeddings]
      .write
      .overwrite()
      .save(savePath)

    AutoGGUFEmbeddings.load(savePath)
  }

}
