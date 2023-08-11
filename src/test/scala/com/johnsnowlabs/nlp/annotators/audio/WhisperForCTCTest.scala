package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AudioAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Using

class WhisperForCTCTest extends AnyFlatSpec {
  lazy val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

  behavior of "WhisperForCTC"

  lazy val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("audio_content")
    .setOutputCol("audio_assembler")

  lazy val rawFloats: Array[Float] = Using({
    val pathToFileWithFloats = "src/test/resources/audio/txt/librispeech_asr_0.txt"
    scala.io.Source.fromFile(pathToFileWithFloats)
  }) { bufferedSource =>
    bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
  }.get

  lazy val processedAudioFloats: Dataset[Row] = Seq(rawFloats).toDF("audio_content")

  // Needs to be added manually
//  lazy val modelPathTf =
//    "whisper/exported/openai/whisper-tiny.en/"

  lazy val modelPathOnnx =
    "onnx/exported_onnx/openai/whisper-tiny"

  lazy val modelOnnx: WhisperForCTC = WhisperForCTC
    .loadSavedModel(modelPathOnnx, ResourceHelper.spark)
    .setInputCols("audio_assembler")
    .setOutputCol("document")

  it should "correctly transform speech to text from already processed audio files" taggedAs SlowTest in {
    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, modelOnnx))

    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

    val expected =
      " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."

    assert(transcribedAudio == expected)
  }

  it should "correctly work with Tokenizer" taggedAs SlowTest in {

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, modelOnnx, token))

    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    val tokens = Annotation.collect(pipelineDF, "token").head.map(_.getResult)

    println(tokens.mkString("Array(\"", "\", \"", "\")"))

    val expectedTokens = Array(
      "Mr",
      ".",
      "Quilter",
      "is",
      "the",
      "apostle",
      "of",
      "the",
      "middle",
      "classes",
      "and",
      "we",
      "are",
      "glad",
      "to",
      "welcome",
      "his",
      "gospel",
      ".")

    tokens.zip(expectedTokens).map { case (token, expected) => assert(token == expected) }

  }

  it should "correctly transcribe speech to text from a different language" taggedAs SlowTest in {

    val modelChangedLang: WhisperForCTC =
      modelOnnx.setLanguage("<|de|>").setTask("<|transcribe|>")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, modelChangedLang))

    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

    val expectedText =
      " Die Kilder ist die Posse der Mittelklasse und wir klären zu den ganzen Kildern."

    assert(transcribedAudio == expectedText)
  }
  it should "correctly transcribe and translate speech to text from a different language" taggedAs SlowTest in {

    val modelChangedLangTask: WhisperForCTC =
      modelOnnx.setLanguage("<|de|>").setTask("<|translate|>")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, modelChangedLangTask))

    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

    val expectedText =
      " Mr. Kfilter is the apostle of the middle classes and we are glad to welcome his gospel."

    assert(transcribedAudio == expectedText)
  }

  it should "transform speech to text with LightPipeline" taggedAs SlowTest in {
    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, modelOnnx, token))

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotate(rawFloats)

    println(result("token"))
    assert(result("audio_assembler").nonEmpty)
    assert(result("document").nonEmpty)
    assert(result("token").nonEmpty)
  }

  it should "transform several speeches to text with LightPipeline" taggedAs SlowTest in {
    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, modelOnnx, token))

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val results = lightPipeline.fullAnnotate(Array(rawFloats, rawFloats))

    results.foreach { result =>
      println(result("token"))
      assert(result("audio_assembler").nonEmpty)
      assert(result("document").nonEmpty)
      assert(result("token").nonEmpty)
    }

  }

  it should "be serializable" taggedAs SlowTest in {

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, modelOnnx))

    val pipelineModel = pipeline.fit(processedAudioFloats)
    pipelineModel.stages.last
      .asInstanceOf[WhisperForCTC]
      .write
      .overwrite()
      .save("./tmp_whisper_model")

    val loadedModel = WhisperForCTC.load("./tmp_whisper_model")
    val newPipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, loadedModel))

    newPipeline
      .fit(processedAudioFloats)
      .transform(processedAudioFloats)
      .select("document")
      .show(10, truncate = false)
  }

  it should "not generate on empty audio" taggedAs SlowTest in {
    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, modelOnnx))

    val data = ResourceHelper.spark.read
      .option("inferSchema", value = true)
      .json("src/test/resources/audio/json/audio_floats.json")
      .select($"float_array".cast("array<float>").alias("audio_content"))

    val pipelineDF = pipeline.fit(data).transform(data)

    val transcribedAudio = Annotation.collect(pipelineDF, "document")

    // Last parsed row of the data has null audio. So the results should be empty.
    val lastRowResult = transcribedAudio.last.head.result
    assert(lastRowResult.isEmpty)

  }

}
