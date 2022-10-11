package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.BufferedSource

class AudioAssemblerTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

  val processedAudioFloats: DataFrame =
    spark.read
      .option("inferSchema", value = true)
      .json("src/test/resources/audio/json/audio_floats.json")
      .select($"float_array".cast("array<float>"))

  processedAudioFloats.printSchema()
  processedAudioFloats.show()

  val bufferedSource: BufferedSource =
    scala.io.Source.fromFile("src/test/resources/audio/csv/audi_floats.csv")

  val rawFloats: Array[Float] = bufferedSource
    .getLines()
    .map(_.split(",").head.trim.toFloat)
    .toArray

  bufferedSource.close

  it should "work with array of doubles" taggedAs FastTest in {

    val processedAudioDoubles: DataFrame =
      spark.read
        .option("inferSchema", value = true)
        .json("src/test/resources/audio/json/audio_floats.json")
        .select($"float_array")

    processedAudioDoubles.printSchema()
    processedAudioDoubles.show()

    val audioAssembler = new AudioAssembler()
      .setInputCol("float_array")
      .setOutputCol("audio_assembler")

    audioAssembler.transform(processedAudioDoubles).collect()

    val bufferedSource: BufferedSource =
      scala.io.Source.fromFile("src/test/resources/audio/csv/audi_floats.csv")

    val rawDoubles: Array[Double] = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toDouble)
      .toArray

    audioAssembler.assemble(rawDoubles, Map("" -> ""))
    audioAssembler.transform(processedAudioDoubles).collect()

  }

  it should "run in a LightPipeline" taggedAs FastTest in {
    val audioAssembler = new AudioAssembler()
      .setInputCol("float_array")
      .setOutputCol("audio_assembler")

    audioAssembler.assemble(rawFloats, Map("" -> ""))
    audioAssembler.transform(processedAudioFloats).collect()
  }

}
