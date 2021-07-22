package com.johnsnowlabs.util

import org.scalatest.FlatSpec

class ConfigLoaderTest extends FlatSpec {

  "ConfigLoader" should "load default property values" ignore {
    val pretrainedS3BucketKey = ConfigHelper.pretrainedS3BucketKey
    val expectedPretrainedS3BucketKeyValue = "auxdata.johnsnowlabs.com"

    val actualPretrainedS3BucketKeyValue = ConfigLoader.getConfigStringValue(pretrainedS3BucketKey)

    assert(expectedPretrainedS3BucketKeyValue == actualPretrainedS3BucketKeyValue)
  }

}
