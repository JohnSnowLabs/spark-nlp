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

package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.util.Version
import org.scalatest.flatspec.AnyFlatSpec

import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.TimeZone

class ResourceMedataTest extends AnyFlatSpec {

  "ResourceMetadata" should "get model with spark version number 3.0 even if version 2.0 has the latest trained date" in {
    val resourcePath = "src/test/resources/resource-downloader/test_v2_latest_date.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "explain_document_dl",
      libVersion = Version(List(3, 3, 2)),
      sparkVersion = Version(List(3, 0, 2)))
    val expectedSparkNLPVersion = Version(List(3, 1, 3))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "take version 3.1.1 when SparkNLP is 3.1.1 and Spark is 2.x" in {
    val resourcePath = "src/test/resources/resource-downloader/test_v2_latest_date.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "explain_document_dl",
      libVersion = Version(List(3, 1, 1)),
      sparkVersion = Version(List(2, 4, 5)))
    val expectedSparkNLPVersion = Version(List(3, 1, 1))
    val expectedSparkVersion = Version(List(2, 4))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark version number 3.0 when spark version is 3.0.1" in {
    val resourcePath = "src/test/resources/resource-downloader/test_with_spark_3.2.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "explain_document_dl",
      libVersion = Version(List(3, 3, 2)),
      sparkVersion = Version(List(3, 0, 1)))
    val expectedSparkNLPVersion = Version(List(3, 1, 3))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark version number 3.0 when spark version is 3.1.0" in {
    val resourcePath = "src/test/resources/resource-downloader/test_with_spark_3.2.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "explain_document_dl",
      libVersion = Version(List(3, 3, 2)),
      sparkVersion = Version(List(3, 1, 0)))
    val expectedSparkNLPVersion = Version(List(3, 1, 3))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark version 3.2 when spark spark version is >= 3.2 " in {
    val resourcePath = "src/test/resources/resource-downloader/test_with_spark_3.2.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "explain_document_dl",
      libVersion = Version(List(3, 4, 0)),
      sparkVersion = Version(List(3, 2, 0)))
    val expectedSparkNLPVersion = Version(List(3, 4, 0))
    val expectedSparkVersion = Version(List(3, 2))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark==3.0 and sparknlp==2.4.5 when spark==3.0 and spark-nlp==2.4.5" in {
    val resourcePath = "src/test/resources/resource-downloader/test_example1.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "bert_base_cased",
      libVersion = Version(List(2, 4, 5)),
      sparkVersion = Version(List(3, 0)))
    val expectedSparkNLPVersion = Version(List(2, 4, 5))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark==3.0 and spark-nlp==3.3.0 when spark==3.0 and spark-nlp==3.3.0" in {
    val resourcePath = "src/test/resources/resource-downloader/test_models_same_time.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "bert_base_cased",
      libVersion = Version(List(3, 3, 0)),
      sparkVersion = Version(List(3, 0)))
    val expectedSparkNLPVersion = Version(List(3, 3, 0))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get model with spark==3.0 and spark-nlp==3.3.0 when spark==3.0 and spark-nlp==3.3.0 and newest model version is 3.0" in {
    val resourcePath = "src/test/resources/resource-downloader/test_bert_v3_newest.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "bert_base_cased",
      libVersion = Version(List(3, 3, 0)),
      sparkVersion = Version(List(3, 0)))
    val expectedSparkNLPVersion = Version(List(3, 3, 0))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)

  }

  it should "get model with spark==3.0 and spark-nlp==3.3.0 when spark==3.0 and spark-nlp==3.3.0 and newest model version is 2.4" in {
    val resourcePath = "src/test/resources/resource-downloader/test_bert_v2_newest.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "bert_base_cased",
      libVersion = Version(List(3, 3, 0)),
      sparkVersion = Version(List(3, 0)))
    val expectedSparkNLPVersion = Version(List(3, 3, 0))
    val expectedSparkVersion = Version(List(3, 0))

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
  }

  it should "get most recent model when spark and spark-nlp versions are the same" in {
    val resourcePath = "src/test/resources/resource-downloader/test_bert_v2_newest.json"
    val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)
    val resourceMetadata = mockResourceDownloader.resources
    val resourceRequest = ResourceRequest(
      "tfhub_use_multi",
      libVersion = Version(List(3, 3, 4)),
      sparkVersion = Version(List(3, 0)))
    val expectedSparkNLPVersion = Version(List(3, 3, 0))
    val expectedSparkVersion = Version(List(3, 0))
    val expectedTimestamp = getTimestamp("2021-05-06T17:52:37.778Z")

    val versions = ResourceMetadata.resolveResource(resourceMetadata, resourceRequest)

    assert(versions.get.sparkVersion.get == expectedSparkVersion)
    assert(versions.get.libVersion.get == expectedSparkNLPVersion)
    assert(versions.get.time == expectedTimestamp)
  }

  private def getTimestamp(date: String): Timestamp = {
    val UTC = TimeZone.getTimeZone("UTC")
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    dateFormat.setTimeZone(UTC)
    val parsedDate = dateFormat.parse(date)
    new Timestamp(parsedDate.getTime)
  }

}
