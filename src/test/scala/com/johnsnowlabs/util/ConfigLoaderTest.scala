package com.johnsnowlabs.util

import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class ConfigLoaderTest extends FlatSpec {

  "ConfigLoader" should "load default property values" in {
    val pretrainedS3BucketKey = ConfigHelperV2.pretrainedS3BucketKey
    val expectedPretrainedS3BucketKeyValue = "auxdata.johnsnowlabs.com"

    val actualPretrainedS3BucketKeyValue = ConfigLoaderV2.getConfigStringValue(pretrainedS3BucketKey)

    assert(expectedPretrainedS3BucketKeyValue == actualPretrainedS3BucketKeyValue)
  }

  it should "load property values from spark session" in {
    SparkSession.getActiveSession.getOrElse(SparkSession.builder()
      .appName("SparkNLP Default Session")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000m")
      .config(ConfigHelperV2.pretrainedS3BucketKey, "custom.value")
      .getOrCreate())
    val pretrainedS3BucketKey = ConfigHelperV2.pretrainedS3BucketKey
    val expectedPretrainedS3BucketKeyValue = "custom.value"

    val actualPretrainedS3BucketKeyValue = ConfigLoaderV2.getConfigStringValue(pretrainedS3BucketKey)

    assert(expectedPretrainedS3BucketKeyValue == actualPretrainedS3BucketKeyValue)
  }

}
