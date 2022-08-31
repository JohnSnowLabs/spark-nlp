package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.audio.AudioAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.sun.media.sound.WaveFileReader
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.net.URL

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

  "AudioAssembler" should "annotate audio data in DataFrame" taggedAs FastTest in {
//    wavDf.show() // Header: 52 49 46 46
//    flacDf.show() // Header: 66 4C 61 43
//
//
//    wavDf.printSchema()
//    root
//     |-- path: string (nullable = true)
//     |-- modificationTime: timestamp (nullable = true)
//     |-- length: long (nullable = true)
//     |-- audio: binary (nullable = true)


    val audioAssembler = new AudioAssembler()
      .setInputCol("audio")
      .setOutputCol("audio_assembler")

    val assembled = audioAssembler.transform(wavDf).collect()
//
//    val result = AssertAnnotations.getActualImageResult(assembled, "image_assembler")
//
    assert(assembled.nonEmpty)
//
//    result.foreach(annotationImages =>
//      annotationImages.foreach { annotationImage =>
//        assert(annotationImage.annotatorType == IMAGE)
//        assert(annotationImage.origin.contains(imagesPath))
//        assert(annotationImage.height >= 0)
//        assert(annotationImage.width >= 0)
//        assert(annotationImage.nChannels >= 0)
//        assert(annotationImage.mode >= 0)
//        assert(annotationImage.result.nonEmpty)
//        assert(annotationImage.metadata.nonEmpty)
//      })

//    Reference: Image file format:
//   root
//    |-- image: struct (nullable = true)
//    |    |-- origin: string (nullable = true)
//    |    |-- height: integer (nullable = true)
//    |    |-- width: integer (nullable = true)
//    |    |-- nChannels: integer (nullable = true)
//    |    |-- mode: integer (nullable = true)
//    |    |-- data: binary (nullable = true)
  }

  it should "support various file formats" taggedAs FastTest in {
    val wavReader = new WaveFileReader()
    val fileFormat = wavReader.getAudioFileFormat(new File(wavPath))

//    val inputStream = wavReader.getAudioInputStream(new URL("file:///" + wavPath))
    print(fileFormat)
  }
  it should "run in a LightPipeline" taggedAs FastTest in {}

}
