package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.sun.media.sound.WaveFileReader
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.net.URL
import scala.io.Source

class AudioAssemblerTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

  val wavPath = "src/test/resources/audio/1272-135031-0014.wav"
  val flacPath = "src/test/resources/audio/1272-135031-0014.flac"

  val wavDf: DataFrame = spark.read
    .format("binaryFile")
    .load(wavPath)
    .withColumnRenamed("content", "audio")

  val flacDf: DataFrame = spark.read
    .format("binaryFile")
    .load(flacPath)
    .withColumnRenamed("content", "audio")

  def almostEqual(a: Float, b: Float, precision: Float): Boolean = Math.abs(a - b) <= precision

  final val ABSOLUTE_PRECISION: Float = 1e-08.toFloat  // TODO

  "AudioAssembler" should "annotate audio data correctly" taggedAs FastTest in {

    val audioAssembler = new AudioAssembler()
      .setInputCol("audio")
      .setOutputCol("audio_assembler")

    val assembled: Array[Float] = audioAssembler
      .transform(wavDf)
      .select("audio_assembler")
      .as[Array[AnnotationAudio]]
      .first()
      .head
      .result

    val referenceFile = Source.fromFile("src/test/resources/audio/1272-135031-0014.npy")
    val referenceArray: Array[Float] = referenceFile.getLines.map(_.toFloat).toArray
    referenceFile.close()

    val differences: Array[Float] = assembled
      .zip(referenceArray)
      .map { case (a, b) => Math.abs(a - b) }

    assembled
      .zip(referenceArray)
      .foreach { case (annotated: Float, actual: Float) =>
        val precision = Math.max(Math.ulp(annotated), Math.ulp(actual))
//        val precision = ABSOLUTE_PRECISION
        assert(
          almostEqual(annotated, actual, precision),
          f"Values are not close enough. $annotated but actually should be $actual. Difference: ${Math
              .abs(annotated - actual)}")
      }
  }

  it should "handle multi channel wav" taggedAs FastTest in {
    val wavReader = new WaveFileReader()
    val fileFormat = wavReader.getAudioFileFormat(new File(wavPath))

    //    val inputStream = wavReader.getAudioInputStream(new URL("file:///" + wavPath))
    print(fileFormat)
  }

  it should "support various file formats" taggedAs FastTest in {
    //    wavDf.show() // Header: 52 49 46 46
    //    flacDf.show() // Header: 66 4C 61 43
    val wavReader = new WaveFileReader()
    val fileFormat = wavReader.getAudioFileFormat(new File(wavPath))

//    val inputStream = wavReader.getAudioInputStream(new URL("file:///" + wavPath))
    print(fileFormat)
  }
  it should "run in a LightPipeline" taggedAs FastTest in {}

}
