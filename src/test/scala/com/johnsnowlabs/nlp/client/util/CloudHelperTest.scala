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

  "CloudHelper" should "parse S3 URIs" taggedAs FastTest in {
    val s3URIs =
      Array("s3a://my.bucket.com/my/S3/path/my_file.tmp", "s3://my.bucket.com/my/S3/path/")
    val expectedOutput =
      Array(("my.bucket.com", "my/S3/path/my_file.tmp"), ("my.bucket.com", "my/S3/path/"))

    s3URIs.zipWithIndex.foreach { case (s3URI, index) =>
      val (actualBucket, actualKey) = CloudHelper.parseS3URI(s3URI)

      val (expectedBucket, expectedKey) = expectedOutput(index)

      assert(actualBucket == expectedBucket)
      assert(actualKey == expectedKey)
    }
  }

  it should "parse GCP URIs" taggedAs FastTest in {
    val s3URIs =
      Array("gs://my.bucket.com/my/GCP/path/my_file.tmp")
    val expectedOutput =
      Array(("my.bucket.com", "my/GCP/path/my_file.tmp"), ("my.bucket.com", "my/GCP/path/"))

    s3URIs.zipWithIndex.foreach { case (s3URI, index) =>
      val (actualBucket, actualKey) = CloudHelper.parseGCPStorageURI(s3URI)

      val (expectedBucket, expectedKey) = expectedOutput(index)

      assert(actualBucket == expectedBucket)
      assert(actualKey == expectedKey)
    }
  }

  it should "parse Azure URIs" taggedAs FastTest in {
    //Azure Blob URI structure is typically: https://[accountName].blob.core.windows.net/[containerName]/[blobName]
    val azureURIs =
      Array("https://storageAccountName.blob.core.windows.net/myblob/path", "https://storageAccountName.blob.core.windows.net/myblob/myfolder/myfile.txt")
    val expectedOutput =
      Array(("myblob", "path"), ("myblob", "myfolder/myfile.txt"))

    azureURIs.zipWithIndex.foreach { case (azureURI, index) =>
      val (actualBucket, actualKey) = CloudHelper.parseAzureBlobURI(azureURI)

      val (expectedBucket, expectedKey) = expectedOutput(index)

      assert(actualBucket == expectedBucket)
      assert(actualKey == expectedKey)
    }
  }

  it should "parse account name from Azure URIs" taggedAs FastTest in {
    val azureURIs =
      Array("https://accountName.blob.core.windows.net", "https://storageAccountName.blob.core.windows.net/myblob/myfolder/myfile.txt")
    val expectedOutput =
      Array("accountName", "storageAccountName")

    azureURIs.zipWithIndex.foreach { case (azureURI, index) =>
      val actualAccountName = CloudHelper.getAccountNameFromAzureBlobURI(azureURI)

      val expectedAccountName = expectedOutput(index)

      assert(actualAccountName == expectedAccountName)
    }
  }

}
