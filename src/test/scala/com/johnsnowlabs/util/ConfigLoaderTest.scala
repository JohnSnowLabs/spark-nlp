package com.johnsnowlabs.util

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class ConfigLoaderTest extends FlatSpec {

  "ConfigLoader" should "load default property values" taggedAs FastTest in {
    val pretrainedS3BucketKey = ConfigHelper.pretrainedS3BucketKey
    val expectedPretrainedS3BucketKeyValue = "auxdata.johnsnowlabs.com"

    val actualPretrainedS3BucketKeyValue = ConfigLoader.getConfigStringValue(pretrainedS3BucketKey)

    assert(expectedPretrainedS3BucketKeyValue == actualPretrainedS3BucketKeyValue)
  }

  "ConfigLoader" should "load property values from spark session" ignore {
    SparkSession.getActiveSession.getOrElse(SparkSession.builder()
      .appName("SparkNLP Default Session")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000m")
      .config(ConfigHelper.pretrainedS3BucketKey, "custom.value")
      .getOrCreate())
    val pretrainedS3BucketKey = ConfigHelper.pretrainedS3BucketKey
    val expectedPretrainedS3BucketKeyValue = "custom.value"

    val actualPretrainedS3BucketKeyValue = ConfigLoader.getConfigStringValue(pretrainedS3BucketKey)

    assert(expectedPretrainedS3BucketKeyValue == actualPretrainedS3BucketKeyValue)
  }

}
