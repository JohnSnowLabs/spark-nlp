package com.johnsnowlabs.nlp.gguf

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.lang.management.ManagementFactory

class AutoGGUFModelTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFModelTest"

  lazy val modelPath =
    "/home/ducha/Workspace/building/java-llama.cpp/models/codellama-7b.Q2_K.gguf"

  lazy val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val model = AutoGGUFModel
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("completions")
    .setBatchSize(4)
    .setNPredict(5)
    .setNGpuLayers(99)

  lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

  it should "create completions" in {
    val data = Seq("Hello, I am a").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }

  it should "create batch completions" in {
    val jvmName = ManagementFactory.getRuntimeMXBean.getName
    val pid = jvmName.split("@")(0)
    println(s"Running in PID $pid")

    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

    val data = Seq(
      "Hello, I am a",
      "The newtonian laws of motion are",
      "A Hohmann transfer is",
      "The most important thing about orbital dynamics is").toDF("text")

    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }

}
