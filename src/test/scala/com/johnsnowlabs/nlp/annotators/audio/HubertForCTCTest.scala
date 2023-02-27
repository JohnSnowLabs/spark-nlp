package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AudioAssembler}
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

class HubertForCTCTest extends AnyFlatSpec {

  val spark: SparkSession = ResourceHelper.spark

  import spark.implicits._

  val pathToFileWithFloats = "src/test/resources/audio/csv/audio_floats.csv"

  val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("audio_content")
    .setOutputCol("audio_assembler")

  val processedAudioFloats: Dataset[Row] =
    spark.read
      .option("inferSchema", value = true)
      .json("src/test/resources/audio/json/audio_floats.json")
      .select($"float_array".cast("array<float>").as("audio_content"))

  // Needs to be added manually
  val modelPath = "src/test/resources/hubert-large-ls960-ft-sparknlp"

  behavior of "HubertForCTC"

  it should "load from saved model" taggedAs SlowTest in {

    val hubert: HubertForCTC = HubertForCTC
      .loadSavedModel(modelPath, spark)
      .setInputCols("audio_assembler")
      .setOutputCol("text")

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, hubert))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    val text = Annotation.collect(pipelineDF, "text").head.head.result
    val expected =
      "MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL "
    assert(text == expected)
  }

  it should "correctly transform speech to text from already processed audio files" taggedAs SlowTest in {

    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("text")

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    Benchmark.measure(iterations = 1, forcePrint = true, description = "Time to show result") {
      pipelineDF.select("text").show(10, truncate = false)
    }

  }

  it should "correctly work with Tokenizer" taggedAs SlowTest in {

    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, speechToText, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    pipelineDF.select("document").show(10, truncate = false)
    pipelineDF.select("token").show(10, truncate = false)

  }

  it should "transform speech to text with LightPipeline" taggedAs SlowTest in {
    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, speechToText, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotate(rawFloats)

    assert(result("audio_assembler").nonEmpty)
    assert(result("document").nonEmpty)
    assert(result("token").nonEmpty)
  }

  it should "transform several speeches to text with LightPipeline" taggedAs SlowTest in {
    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, speechToText, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val results = lightPipeline.fullAnnotate(Array(rawFloats, rawFloats))

    results.foreach { result =>
      assert(result("audio_assembler").nonEmpty)
      assert(result("document").nonEmpty)
      assert(result("token").nonEmpty)
    }

  }

  it should "be serializable" taggedAs SlowTest in {

    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("text")

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

    val pipelineModel = pipeline.fit(processedAudioFloats)
    pipelineModel.stages.last
      .asInstanceOf[HubertForCTC]
      .write
      .overwrite()
      .save("./tmp_hubert_model")

    val loadedModel = HubertForCTC.load("./tmp_hubert_model")
    val newPipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, loadedModel))

    newPipeline
      .fit(processedAudioFloats)
      .transform(processedAudioFloats)
      .select("text")
      .show(10, truncate = false)

  }

  it should "benchmark" taggedAs SlowTest in {

    val speechToText = HubertForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("text")

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

    Array(1, 2, 4, 8).foreach(b => {
      speechToText.setBatchSize(b)

      val pipelineModel = pipeline.fit(processedAudioFloats)
      val pipelineDF = pipelineModel.transform(processedAudioFloats)

      println(s"batch size: ${pipelineModel.stages.last.asInstanceOf[HubertForCTC].getBatchSize}")

      Benchmark.measure(
        iterations = 1,
        forcePrint = true,
        description = "Time to save pipeline") {
        pipelineDF.select("text").count()
      }
    })
  }

  //  it should "pretrained pipeline" taggedAs SlowTest in {
  //
  //    val processedAudioDoubles: Dataset[Row] =
  //      spark.read
  //        .option("inferSchema", value = true)
  //        .json("src/test/resources/audio/json/audio_floats.json")
  //        .select($"float_array".as("audio_content"))
  //
  //    processedAudioDoubles.printSchema()
  //
  //    val pipelineModel = PretrainedPipeline("pipeline_asr_hubert_large_ls960_ctc")
  //
  //    val pipelineDF = pipelineModel.transform(processedAudioDoubles)
  //    pipelineDF.show()
  //
  //  }

}
