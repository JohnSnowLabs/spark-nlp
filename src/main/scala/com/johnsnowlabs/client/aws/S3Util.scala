/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.client.aws

object S3Util {

  def extractBucketAndKeyPrefixFromS3Path(s3Path: String): (String, String) = {
    val s3Bucket = s3Path.replace("s3://", "").split("/").head
    val keyPrefix = s3Path.substring(s"s3://$s3Bucket".length + 1)

    (s3Bucket, keyPrefix)
  }

  def buildS3FilePath(s3Path: String, sourceFilePath: String): String = {
    val (_, s3FilePathPrefix) = S3Util.extractBucketAndKeyPrefixFromS3Path(s3Path)
    val fileName = sourceFilePath.split("/").last
    val s3FilePath =
      if (s3Path.last == '/') s"$s3FilePathPrefix$fileName" else s"$s3FilePathPrefix/$fileName"

    s3FilePath
  }

  def getS3File(parts: String*): String = {
    parts
      .map(part => part.stripSuffix("/"))
      .filter(part => part.nonEmpty)
      .mkString("/")
  }

}
