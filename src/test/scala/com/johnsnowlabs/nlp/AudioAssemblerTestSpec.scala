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

  it should "run in a LightPipeline" taggedAs FastTest in {
    val audioAssembler = new AudioAssembler()
      .setInputCol("float_array")
      .setOutputCol("audio_assembler")

    audioAssembler.assemble(rawFloats, Map("" -> ""))
    audioAssembler.transform(processedAudioFloats).collect()
  }

}
