package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.Version
import org.scalatest.FlatSpec

import java.sql.Timestamp


class ResourceDownloaderSpec extends FlatSpec {
  val b = CloudTestResources

  "CloudResourceMetadata" should "serialize and deserialize correctly" taggedAs FastTest in {
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

  "CloudResourceDownloader" should "choose the newest versions" taggedAs FastTest in {
    val found = ResourceMetadata.resolveResource(b.all, ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isDefined)
    assert(found.get == b.name_en_123_345_new)
  }

  "CloudResourceDownloader" should "filter disabled resources" taggedAs FastTest in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_new_disabled), ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isEmpty)
  }

  "CloudResourceDownloader" should "filter language and allow empty versions" taggedAs FastTest in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_old, b.name_de), ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isDefined)
    assert(found.get == b.name_en_old)
  }

  "CloudResourceDownloader" should "allow download of model for 2.4 for 2.3 found resource" in{
    val found = ResourceMetadata.resolveResource(List(b.name_en_251_23), ResourceRequest("name", Some("en"), "", Version(2, 5, 1), Version(2, 4, 4)))
    assert(found.isDefined)
  }

  "CloudResourceDownloader" should "not allow download of model for 3 for 2.3 found resource" in{
    val found = ResourceMetadata.resolveResource(List(b.name_en_251_23), ResourceRequest("name", Some("en"), "", Version(2, 5, 1), Version(3)))
    assert(found.isEmpty)
  }

  "CloudResourceDownloader" should "allow download of model for 3.0.x on spark 3.x found resource" in{
    val found = ResourceMetadata.resolveResource(List(b.name_en_300_30), ResourceRequest("name", Some("en"), "", Version(3, 0, 0), Version(3, 0)))
    assert(found.isDefined)
  }

  "Pretrained" should "allow download of BERT Tiny from public S3 Bucket" in{
    BertEmbeddings.pretrained("small_bert_L2_128", lang = "en")
  }

  "Pretrained" should "allow download of BERT Tiny from community S3 Bucket" in{
    BertEmbeddings.pretrained("small_bert_L2_128_test", lang = "en", remoteLoc = "@maziyarpanahi")
  }

  "Pretrained" should "allow download of BERT Tiny from public and community S3 Buckets" in{
    BertEmbeddings.pretrained("small_bert_L2_128_test", lang = "en", remoteLoc = "@maziyarpanahi")
    BertEmbeddings.pretrained("small_bert_L2_128", lang = "en")
  }

}