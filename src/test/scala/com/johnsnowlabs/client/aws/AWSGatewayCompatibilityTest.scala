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

package com.johnsnowlabs.client.aws

import com.amazonaws.AmazonServiceException
import com.amazonaws.services.s3.model.ObjectMetadata
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.nio.file.Files

class AWSGatewayCompatibilityTest extends AnyFlatSpec {

  private val MiB = 1024 * 1024
  private val Key = "public/models/fake_model_en_7.0.0_3.0_1.zip"

  private def target(): File =
    File.createTempFile("model", ".zip", Files.createTempDirectory("compat").toFile)

  "getS3Object(bucket, key, file)" should "still return the object's metadata and fill the file" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    val file = target()
    val metadata: ObjectMetadata =
      FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, file)

    assert(metadata.getContentLength == 16L * MiB)
    assert(Files.readAllBytes(file.toPath) sameElements s3.bytes(Key))
  }

  it should "raise S3's 404 for a missing object, as before" taggedAs FastTest in {
    val error = intercept[AmazonServiceException] {
      FakeS3.gatewayFor(new FakeS3(), threads = 4).getS3Object("bucket", "missing.zip", target())
    }
    assert(error.getStatusCode == 404)
  }

  it should "not publish progress, as in 7.0" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    val before = DownloadProgressTracker.snapshotWithGeneration.split('|')(0)
    FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, target())

    assert(
      DownloadProgressTracker.snapshotWithGeneration.split('|')(0) == before,
      "callers built against 7.0 never saw a progress display from this method")
  }

  "doesS3ObjectExist" should "answer from a HEAD" taggedAs FastTest in {
    val gateway = FakeS3.gatewayFor(new FakeS3().put(Key, Array[Byte](1, 2, 3)), threads = 4)
    assert(gateway.doesS3ObjectExist("bucket", Key))
    assert(!gateway.doesS3ObjectExist("bucket", "missing.zip"))
  }
}
