package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.TestUtils.measureRAMChange
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFModelTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFModelTest"

  lazy val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val model: AutoGGUFModel = AutoGGUFModel
    .pretrained()
    .setInputCols("document")
    .setOutputCol("completions")
    .setBatchSize(4)
    .setNPredict(20)
    .setNGpuLayers(99)
    .setTemperature(0.4f)
    .setNCtx(512)
    .setTopK(40)
    .setTopP(0.9f)
    .setPenalizeNl(true)

  lazy val data: Dataset[Row] = Seq(
    "The moons of Jupiter are ", // "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones. The four"
    "Earth is ", // "Earth is 4.5 billion years old. It has been home to countless species, some of which have gone extinct, while others have evolved into"
    "The moon is ", // "The moon is 1/400th the size of the sun. The sun is 1.39 million kilometers in diameter, while"
    "The sun is " //
  ).toDF("text").repartition(1)

  lazy val pipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, model))

  def assertAnnotationsNonEmpty(resultDf: DataFrame): Unit = {
    Annotation
      .collect(resultDf, "completions")
      .foreach(annotations => {
        println(annotations.head)
        assert(annotations.head.result.nonEmpty)
      })
  }

  it should "create completions" taggedAs SlowTest in {
    val data = Seq("Hello, I am a").toDF("text")
    val result = pipeline.fit(data).transform(data)
    assertAnnotationsNonEmpty(result)
  }

  it should "create batch completions" taggedAs SlowTest in {
    val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)
    assertAnnotationsNonEmpty(result)
  }

  it should "be serializable" taggedAs SlowTest in {
    val data = Seq("Hello, I am a").toDF("text")
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    model.setNPredict(5)

    val pipelineModel = pipeline.fit(data)
    val savePath = "./tmp_autogguf_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFModel]
      .write
      .overwrite()
      .save(savePath)

    val loadedModel = AutoGGUFModel.load(savePath)
    val newPipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, loadedModel))

    newPipeline
      .fit(data)
      .transform(data)
      .select("completions")
      .show(truncate = false)
  }

  it should "accept all parameters that are settable" taggedAs SlowTest in {
    // Model Parameters
    model.setNThreads(8)
//    model.setNThreadsDraft(8)
    model.setNThreadsBatch(8)
//    model.setNThreadsBatchDraft(8)
    model.setNCtx(512)
    model.setNBatch(32)
    model.setNUbatch(32)
    model.setNDraft(5)
    model.setLogVerbosity(0)
    model.setDisableLog(true)
//    model.setNChunks(-1)
//    model.setNSequences(1)
//    model.setPSplit(0.1f)
    model.setNGpuLayers(99)
    model.setNGpuLayersDraft(99)
    model.setGpuSplitMode("NONE")
    model.setMainGpu(0)
//    model.setTensorSplit(Array[Double]())
//    model.setGrpAttnN(1)
//    model.setGrpAttnW(512)
    model.setRopeFreqBase(1.0f)
    model.setRopeFreqScale(1.0f)
    model.setYarnExtFactor(1.0f)
    model.setYarnAttnFactor(1.0f)
    model.setYarnBetaFast(32.0f)
    model.setYarnBetaSlow(1.0f)
    model.setYarnOrigCtx(0)
    model.setDefragmentationThreshold(-1.0f)
    model.setNumaStrategy("DISTRIBUTE")
    model.setRopeScalingType("NONE")
    model.setModelDraft("")
//    model.setLookupCacheStaticFilePath("/tmp/sparknlp-llama-cpp-cache")
//    model.setLookupCacheDynamicFilePath("/tmp/sparknlp-llama-cpp-cache")
    model.setFlashAttention(false)
//    model.setInputPrefixBos(false)
    model.setUseMmap(false)
    model.setUseMlock(false)
    model.setNoKvOffload(false)
    model.setSystemPrompt("")
    model.setChatTemplate("")

    // Inference Parameters
    model.setInputPrefix("")
    model.setInputSuffix("")
    model.setCachePrompt(true)
    model.setNPredict(-1)
    model.setTopK(40)
    model.setTopP(0.9f)
    model.setMinP(0.1f)
    model.setTfsZ(1.0f)
    model.setTypicalP(1.0f)
    model.setTemperature(0.8f)
    model.setDynamicTemperatureRange(0.0f)
    model.setDynamicTemperatureExponent(1.0f)
    model.setRepeatLastN(64)
    model.setRepeatPenalty(1.0f)
    model.setFrequencyPenalty(0.0f)
    model.setPresencePenalty(0.0f)
    model.setMiroStat("DISABLED")
    model.setMiroStatTau(5.0f)
    model.setMiroStatEta(0.1f)
    model.setPenalizeNl(false)
    model.setNKeep(0)
    model.setSeed(-1)
    model.setNProbs(0)
    model.setMinKeep(0)
    model.setGrammar("")
    model.setPenaltyPrompt("")
    model.setIgnoreEos(false)
    model.setDisableTokenIds(Array[Int]())
    model.setStopStrings(Array[String]())
    model.setUseChatTemplate(false)
    model.setNPredict(2)
    model.setSamplers(Array("TOP_P", "TOP_K"))

    // Struct Features
    model.setTokenIdBias(Map(0 -> 0.0f, 1 -> 0.0f))
    model.setTokenBias(Map("!" -> 0.0f, "?" -> 0.0f))
//    model.setLoraAdapters(Map(" " -> 0.0f))

    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }

  it should "contain metadata when loadSavedModel" taggedAs SlowTest in {
    lazy val modelPath = "models/codellama-7b.Q2_K.gguf"
    val model = AutoGGUFModel.loadSavedModel(modelPath, ResourceHelper.spark)
    val metadata = model.getMetadata
    assert(metadata.nonEmpty)

    val metadataMap = model.getMetadataMap
    assert(metadataMap.nonEmpty)
  }

  it should "return error messages when completions can't be produced" taggedAs SlowTest in {
    val model = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("completions")
      .setGrammar("root ::= (") // Invalid grammar

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)

    val collected = Annotation
      .collect(result, "completions")

    assert(collected.length == data.count().toInt, "Should return the same number of rows")
    collected
      .foreach(annotations => {
        assert(annotations.head.result.isEmpty, "Completions should be empty")
        assert(
          annotations.head.metadata.contains("llamacpp_exception"),
          "llamacpp_exception should be present")
      })
  }

  it should "be able to also load pretrained AutoGGUFVisionModels" taggedAs SlowTest in {
    val model = AutoGGUFModel
      .pretrained("Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf")
      .setInputCols("document")
      .setOutputCol("completions")
      .setBatchSize(2)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)

    result.show()
  }

  it should "accept protocol prepended paths" taggedAs SlowTest in {
    val data = Seq("Hello, I am a").toDF("text")
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    val pipelineModel = pipeline.fit(data)

    val savePath = "file:///tmp/tmp_autogguf_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFModel]
      .write
      .overwrite()
      .save(savePath)

    AutoGGUFModel.load(savePath)
  }

  // This test requires cpu
  it should "be closeable" taggedAs SlowTest in {
    val model = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("completions")

    val data = Seq("Hello, I am a").toDF("text")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    pipeline.fit(data).transform(data).show()

    val ramChange = measureRAMChange { model.close() }
    println("Freed RAM after closing the model: " + ramChange + " MB")
    assert(ramChange < -100, "Freed RAM should be greater than 100 MB")
  }

//  it should "benchmark" taggedAs SlowTest in {
//    val model = AutoGGUFModel
//      .loadSavedModel("models/gemma-3-4b-it-qat-Q4_K_M.gguf", ResourceHelper.spark)
//      .setInputCols("document")
//      .setOutputCol("completions")
//      .setNPredict(100)
//      .setBatchSize(8)
//      .setNGpuLayers(99)
//
//    val benchmarkData =
//      Seq.fill(200)("All work and no play makes Jack a dull boy.").toDF("text").repartition(4)
//
//    val pipeline =
//      new Pipeline().setStages(Array(documentAssembler, model))
//
//    Benchmark.measure("Batch benchmark") {
//      val result = pipeline.fit(benchmarkData).transform(benchmarkData)
//      val collected = Annotation.collect(result, "completions")
//      assert(collected.nonEmpty, "Completions should not be empty")
//    }
//  }
//
//
//  it should "be compatible with sentencesplitter" taggedAs SlowTest in {
//    // TODO
//    val model = AutoGGUFModel
//      .pretrained()
//      .setInputCols("document")
//      .setOutputCol("completions")
//
//    val pipeline =
//      new Pipeline().setStages(Array(documentAssembler, model))
//    val result = pipeline.fit(data).transform(data)
//
//    val collected = Annotation
//      .collect(result, "completions")
//
//    assert(collected.length == data.count().toInt, "Should return the same number of rows")
//    collected
//      .foreach(annotations => {
//        assert(annotations.head.result.isEmpty, "Completions should be empty")
//        assert(
//          annotations.head.metadata.contains("llamacpp_exception"),
//          "llamacpp_exception should be present")
//      })
//  }
}
