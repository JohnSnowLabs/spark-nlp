package com.johnsnowlabs.nlp.pretrained

/***
 * Mock class of the ResourceDownloader, that will use a local file instead of downloading the metadata.json from
 * S3.
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

}
