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

import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.{TrainingHelper, Version}
import org.scalatest.flatspec.AnyFlatSpec

import java.sql.Timestamp

class ResourceDownloaderSpec extends AnyFlatSpec {
  val b: CloudTestResources.type = CloudTestResources

  "CloudResourceMetadata" should "serialize and deserialize correctly" taggedAs FastTest in {
    val resource = new ResourceMetadata(
      "name",
      Some("en"),
      Some(Version(1, 2, 3)),
      Some(Version(5, 4, 3)),
      true,
      new Timestamp(123213))

    val json = ResourceMetadata.toJson(resource)
    val deserialized = ResourceMetadata.parseJson(json)

    assert(deserialized == resource)
  }

  "CloudResourceDownloader" should "choose the newest Spark version" taggedAs FastTest in {
    val sparkNLPVersion = Version(1, 2, 3)
    val sparkVersion = Version(3, 4, 5)
    val found = ResourceMetadata.resolveResource(
      b.all,
      ResourceRequest("name", Some("en"), "", sparkNLPVersion, sparkVersion))

    assert(found.isDefined)
    assert(found.get == b.name_en_123_345_new)
  }

  "CloudResourceDownloader" should "choose the newest SparkNLP Version when Spark versions are the same" taggedAs FastTest in {
    val sparkNLPVersion = Version(1, 2, 3)
    val sparkVersion = Version(3, 4)
    val found = ResourceMetadata.resolveResource(
      b.all,
      ResourceRequest("name", Some("en"), "", sparkNLPVersion, sparkVersion))

    assert(found.isDefined)
    assert(found.get == b.name_en_123_34_new)
  }

  "CloudResourceDownloader" should "filter disabled resources" taggedAs FastTest in {
    val found = ResourceMetadata.resolveResource(
      List(b.name_en_new_disabled),
      ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isEmpty)
  }

  "CloudResourceDownloader" should "filter language and allow empty versions" taggedAs FastTest ignore {
    val found = ResourceMetadata.resolveResource(
      List(b.name_en_old, b.name_de),
      ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isDefined)
    assert(found.get == b.name_en_old)
  }

  "CloudResourceDownloader" should "allow download of model for 3.0.x on spark 3.x found resource" in {
    val found = ResourceMetadata.resolveResource(
      List(b.name_en_300_30),
      ResourceRequest("name", Some("en"), "", Version(3, 0, 0), Version(3, 0)))
    assert(found.isDefined)
  }

  "Pretrained" should "allow download of BERT Tiny from public S3 Bucket" in {
    BertEmbeddings.pretrained("small_bert_L2_128", lang = "en")
  }

  "Pretrained" should "allow download of BERT Tiny from community S3 Bucket" in {
    BertEmbeddings.pretrained("small_bert_L2_128_test", lang = "en", remoteLoc = "@maziyarpanahi")
  }

  "Pretrained" should "allow download of BERT Tiny from public and community S3 Buckets" in {
    val bertModel = BertEmbeddings.pretrained(
      "small_bert_L2_128_test",
      lang = "en",
      remoteLoc = "@maziyarpanahi")
    BertEmbeddings.pretrained("small_bert_L2_128", lang = "en")

    TrainingHelper.saveModel(
      name = "tmp_small_bert",
      language = Some("en"),
      libVersion = Some(Version(3, 4, 0)),
      sparkVersion = Some(Version(3, 0)),
      modelWriter = bertModel.write,
      folder = "./tmp_model",
      category = Some(ResourceType.MODEL))
  }

}
