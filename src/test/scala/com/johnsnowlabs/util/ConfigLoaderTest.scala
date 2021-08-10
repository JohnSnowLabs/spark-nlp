/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
