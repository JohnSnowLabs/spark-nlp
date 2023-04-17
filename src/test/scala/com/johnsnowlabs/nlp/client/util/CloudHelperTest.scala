/*
 * Copyright 2017-2023 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.client.util

import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class CloudHelperTest extends AnyFlatSpec {

  it should "parse S3 URIs" taggedAs FastTest in {
    val s3URIs =
      Array("s3a://my.bucket.com/my/S3/path/my_file.tmp", "s3://my.bucket.com/my/S3/path/")
    val expectedOutput =
      Array(("my.bucket.com", "my/S3/path/my_file.tmp"), ("my.bucket.com", "my/S3/path/"))

    s3URIs.zipWithIndex.foreach { case (s3URI, index) =>
      val (actualBucket, actualKey) = CloudHelper.parseS3URI(s3URI)

      val (expectedBucket, expectedKey) = expectedOutput(index)

      assert(expectedBucket == actualBucket)
      assert(expectedKey == actualKey)
    }
  }

}
