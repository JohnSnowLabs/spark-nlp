package com.johnsnowlabs.pretrained

import java.sql.Timestamp
import com.johnsnowlabs.util.Version
import org.scalatest.FlatSpec


class ResourceDownloaderSpec extends FlatSpec {
  val b = S3TestResources

  "S3ResourceMetadata" should "serialize and deserialize correctly" in {
    val resource = new ResourceMetadata("name",
      Some("en"),
      Some(Version(1,2,3)),
      Some(Version(5,4,3)),
      true,
      new Timestamp(123213))

    val json = ResourceMetadata.toJson(resource)
    val deserialized = ResourceMetadata.parseJson(json)

    assert(deserialized == resource)
  }

  "S3ResourceDownloader" should "choose the newest versions" in {
    val found = ResourceMetadata.resolveResource(b.all, "name", Some("en"), Version(1, 2, 3), Version(3, 4, 5))

    assert(found.isDefined)
    assert(found.get == b.name_en_123_345_new)
  }

  "S3ResourceDownloader" should "filter disabled resources" in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_new_disabled), "name", Some("en"), Version(1, 2, 3), Version(3, 4, 5))

    assert(found.isEmpty)
  }

  "S3ResourceDownloader" should "filter language and allow empty versions" in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_old, b.name_de), "name", Some("en"), Version(1, 2, 3), Version(3, 4, 5))

    assert(found.isDefined)
    assert(found.get == b.name_en_old)
  }
}