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

/** * Mock class of the ResourceDownloader, that will use a local file instead of downloading the
  * metadata.json from S3.
  */
class MockResourceDownloader(resourcePath: String) extends ResourceDownloader {

  def download(request: ResourceRequest): Option[String] = {
    Some("MockResourceDownloader")
  }

  def getDownloadSize(request: ResourceRequest): Option[Long] = {
    Some(0)
  }

  def clearCache(request: ResourceRequest): Unit = {}

  val resources: List[ResourceMetadata] = ResourceMetadata.readResources(resourcePath)

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata] = resources
  override def downloadAndUnzipFile(s3FilePath: String, unzip: Boolean = true): Option[String] =
    Some("model")
}
