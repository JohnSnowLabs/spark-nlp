/*
 * Copyright 2017-2026 John Snow Labs
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

package com.johnsnowlabs.nlp.pretrained

import com.amazonaws.services.s3.model.ObjectMetadata
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.Date

class MetadataRefreshTest extends AnyFlatSpec {

  private val index = Paths.get("src/test/resources/resource-downloader/test_metadata.json")

  /** Counts HEADs and GETs of metadata.json; S3's Last-Modified is whatever the test sets. */
  private class CountingGateway
      extends AWSGateway(
        accessKeyId = "",
        secretAccessKey = "",
        sessionToken = "",
        awsProfile = "",
        region = "us-east-1",
        credentialsType = "public") {
    @volatile var lastModifiedInS3: Date = new Date(1000000L)
    @volatile var heads = 0
    @volatile var gets = 0

    override def getS3ObjectMetadata(
        bucket: String,
        s3FilePath: String): Option[ObjectMetadata] = {
      heads += 1
      val metadata = new ObjectMetadata()
      metadata.setLastModified(lastModifiedInS3)
      metadata.setContentLength(Files.size(index))
      Some(metadata)
    }

    override def getS3Object(
        bucket: String,
        s3FilePath: String,
        tmpFile: File,
        objectMetadata: Option[ObjectMetadata],
        showProgress: Boolean): Unit = {
      gets += 1
      Files.copy(index, tmpFile.toPath, StandardCopyOption.REPLACE_EXISTING)
    }
  }

  private def downloaderWith(gateway: AWSGateway): S3ResourceDownloader = {
    val cache = Files.createTempDirectory("metadata-refresh-cache").toString
    new S3ResourceDownloader("bucket", "public/models", cache, "public") {
      override lazy val awsGateway: AWSGateway = gateway
    }
  }

  "needToRefresh" should "refresh only an empty or older cached index" taggedAs FastTest in {
    val inS3 = new Date(2000000L)
    assert(S3ResourceDownloader.needToRefresh(None, inS3))
    assert(S3ResourceDownloader.needToRefresh(Some(new Date(1000000L)), inS3))
    assert(!S3ResourceDownloader.needToRefresh(Some(new Date(2000000L)), inS3))
    assert(!S3ResourceDownloader.needToRefresh(Some(new Date(3000000L)), inS3))
  }

  "downloadMetadataIfNeed" should "check freshness with a HEAD and fetch the index only when it changed" taggedAs FastTest in {
    val gateway = new CountingGateway
    val downloader = downloaderWith(gateway)

    val first = downloader.downloadMetadataIfNeed("public/models")
    assert(first.nonEmpty)
    assert((gateway.heads, gateway.gets) == ((1, 1)), "empty cache: one HEAD, one GET")

    val second = downloader.downloadMetadataIfNeed("public/models")
    assert(second == first)
    assert((gateway.heads, gateway.gets) == ((2, 1)), "equal timestamp: HEAD only")

    gateway.lastModifiedInS3 = new Date(500000L)
    downloader.downloadMetadataIfNeed("public/models")
    assert((gateway.heads, gateway.gets) == ((3, 1)), "S3 older than the cache: HEAD only")

    gateway.lastModifiedInS3 = new Date(9000000L)
    downloader.downloadMetadataIfNeed("public/models")
    assert((gateway.heads, gateway.gets) == ((4, 2)), "S3 newer than the cache: HEAD and GET")
  }
}
