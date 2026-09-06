package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType, AudioAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source
import scala.util.Using

class WhisperForCTCTest extends AnyFlatSpec with WhisperForCTCBehaviors {
  import spark.implicits._

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
  lazy val modelTf: WhisperForCTC = WhisperForCTC
    .pretrained("asr_whisper_tiny", "xx")
    .setInputCols("audio_assembler")
    .setOutputCol("document")

  lazy val modelOnnx: WhisperForCTC = WhisperForCTC
    .pretrained()
    .setInputCols("audio_assembler")
    .setOutputCol("document")

  behavior of "WhisperForCTC"

  it should behave like correctTranscriber(modelTf, "tf")
  it should behave like compatibleWithLightPipeline(modelTf, "tf")
  it should behave like serializableModel(modelTf, "tf")

  it should behave like correctTranscriber(modelOnnx, "onnx")
  it should behave like compatibleWithLightPipeline(modelOnnx, "onnx")
  it should behave like serializableModel(modelOnnx, "onnx")

}

trait WhisperForCTCBehaviors { this: AnyFlatSpec =>
  lazy val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

  val audioAssembler: AudioAssembler
  val processedAudioFloats: Dataset[Row]
  val rawFloats: Array[Float]

  def correctTranscriber(model: => WhisperForCTC, engine: => String): Unit = {
    it should s"correctly transform speech to text from already processed audio files ($engine)" taggedAs SlowTest in {
      val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, model))

      processedAudioFloats.printSchema()

      val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

      val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

      val expected =
        " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."

      assert(transcribedAudio == expected)
    }

    it should s"correctly transcribe batches ($engine)" taggedAs SlowTest in {
      val batchAudioAnnotations =
        Seq(
          Array(rawFloats, rawFloats).map(new AnnotationAudio(AnnotatorType.AUDIO, _, Map.empty)))

      val result: Seq[Annotation] = model.batchAnnotate(batchAudioAnnotations).head

      val expected =
        " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."

      result.foreach(transcription => assert(transcription.getResult == expected))

    }

    it should s"correctly work with Tokenizer ($engine)" taggedAs SlowTest in {

      val token = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")

      val pipeline: Pipeline =
        new Pipeline().setStages(Array(audioAssembler, model, token))

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

    it should s"correctly transcribe speech to text from a different language ($engine)" taggedAs SlowTest in {

      val modelChangedLang: WhisperForCTC =
        model.setLanguage("<|de|>").setTask("<|transcribe|>")

      val pipeline: Pipeline =
        new Pipeline().setStages(Array(audioAssembler, modelChangedLang))

      processedAudioFloats.printSchema()

      val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

      val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

      val expectedText =
        " Die Kilder ist die Posse der Mittelklasse und wir klären zu den ganzen Kildern."

      assert(transcribedAudio == expectedText)
    }

    it should s"correctly transcribe and translate speech to text from a different language ($engine)" taggedAs SlowTest in {

      val modelChangedLangTask: WhisperForCTC =
        model.setLanguage("<|de|>").setTask("<|translate|>")

      val pipeline: Pipeline =
        new Pipeline().setStages(Array(audioAssembler, modelChangedLangTask))

      processedAudioFloats.printSchema()

      val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

      val transcribedAudio = Annotation.collect(pipelineDF, "document").head.head.getResult

      val expectedText =
        " Mr. Kfilter is the apostle of the middle classes and we are glad to welcome his gospel."

      assert(transcribedAudio == expectedText)
    }

    it should s"not generate on empty audio ($engine)" taggedAs SlowTest in {
      val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, model))

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

  def compatibleWithLightPipeline(model: => WhisperForCTC, engine: => String): Unit = {

    it should s"transform speech to text with LightPipeline ($engine)" taggedAs SlowTest in {
      val token = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")

      val pipeline: Pipeline =
        new Pipeline().setStages(Array(audioAssembler, model, token))

      val pipelineModel = pipeline.fit(processedAudioFloats)
      val lightPipeline = new LightPipeline(pipelineModel)
      val result = lightPipeline.fullAnnotate(rawFloats)

      println(result("token"))
      assert(result("audio_assembler").nonEmpty)
      assert(result("document").nonEmpty)
      assert(result("token").nonEmpty)
    }

    it should s"transform several speeches to text with LightPipeline ($engine)" taggedAs SlowTest in {
      val token = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")

      val pipeline: Pipeline =
        new Pipeline().setStages(Array(audioAssembler, model, token))

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
  }
  def serializableModel(model: => WhisperForCTC, engine: => String): Unit = {
    it should s"be serializable ($engine)" taggedAs SlowTest in {

      val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, model))

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
  }
}

class WhisperForCTCRepetitionParamsSpec extends AnyFlatSpec {

  private val scratch = sys.env.getOrElse(
    "SPARKNLP_SPEAKER_DIARIZER_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-abdullah-Documents-spark-nlp--claude-worktrees-opt-125m-research-90713f/61bcb16c-f914-49ff-af4a-c56c4c5e7ac6/scratchpad")
  private val whisperPkgPath = s"$scratch/models/whisper_pkg"
  private val td = s"$scratch/models/testdata"
  private val spark = ResourceHelper.spark

  private def loadFloats(path: String): Array[Float] = {
    val src = Source.fromFile(path)
    try {
      src.getLines().map(_.trim.toFloat).toArray
    } finally src.close()
  }

  private def audio(name: String): Array[Float] = loadFloats(s"$td/${name}_floats.txt")

  private def freshModel(): WhisperForCTC =
    WhisperForCTC
      .loadSavedModel(whisperPkgPath, spark)
      .setInputCols("audio_assembler")
      .setOutputCol("document")

  private def run(model: WhisperForCTC, samples: Array[Float]): Seq[Annotation] = {
    val row = Array(AnnotationAudio(AnnotatorType.AUDIO, samples, Map.empty))
    model.batchAnnotate(Seq(row)).head
  }

  private def text(anns: Seq[Annotation]): String = anns.map(_.result).mkString(" ")

  private lazy val repeatedSpeech: Array[Float] = {
    val clip = audio("single_speaker")
    clip ++ clip
  }

  "WhisperForCTC.setRepetitionPenalty" should "change greedy output on speech that verbatim-repeats itself" taggedAs SlowTest in {
    val baseline = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    val penalized = run(freshModel().setRepetitionPenalty(1.8), repeatedSpeech)

    println(s"[WhisperForCTC repetitionPenalty] baseline=${text(baseline)}")
    println(s"[WhisperForCTC repetitionPenalty] penalized=${text(penalized)}")
    assert(text(baseline).nonEmpty && text(penalized).nonEmpty)
    assert(
      text(baseline) != text(penalized),
      "the same fix verified through SpeakerDiarizer must also take effect through " +
        "WhisperForCTC's own batchAnnotate path, since both share Whisper.getLogitProcessors")
  }

  it should "leave output unchanged at the default penalty of 1.0" taggedAs SlowTest in {
    val a = run(freshModel(), repeatedSpeech)
    val b = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    assert(text(a) == text(b))
  }

  "repetitionPenalty/noRepeatNgramSize" should "survive a WhisperForCTC save/load round trip" taggedAs SlowTest in {
    val original = freshModel().setRepetitionPenalty(1.8).setNoRepeatNgramSize(3)
    val path = s"$scratch/models/save_test_whisperctc_repetition_params"
    original.write.overwrite().save(path)
    val reloaded = WhisperForCTC.load(path)

    assert(reloaded.getRepetitionPenalty == 1.8)
    assert(reloaded.getNoRepeatNgramSize == 3)

    val reloadedResult =
      run(reloaded.setInputCols("audio_assembler").setOutputCol("document"), repeatedSpeech)
    val baseline = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    println(s"[WhisperForCTC save/load] reloaded=${text(reloadedResult)}")
    assert(
      text(reloadedResult) != text(baseline),
      "the reloaded model must actually apply the restored repetitionPenalty during inference, " +
        "not just report the right value from a getter")
  }
}
