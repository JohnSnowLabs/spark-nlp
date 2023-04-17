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
package com.johnsnowlabs.client.util

object CloudHelper {

  def parseS3URI(s3URI: String): (String, String) = {
    val prefix = if (s3URI.startsWith("s3:")) "s3://" else "s3a://"
    val bucketName = s3URI.substring(prefix.length).split("/").head
    val key = s3URI.substring((prefix + bucketName).length + 1)

    require(bucketName.nonEmpty, "S3 bucket name is empty!")

    (bucketName, key)
  }

  def parseGCPStorageURI(gcpStorageURI: String): (String, String) = {
    val prefix = "gs://"
    val bucketName = gcpStorageURI.substring(prefix.length).split("/").head
    val storagePath = gcpStorageURI.substring((prefix + bucketName).length + 1)

    require(bucketName.nonEmpty, "GCP Storage bucket name is empty!")

    (bucketName, storagePath)
  }

}
