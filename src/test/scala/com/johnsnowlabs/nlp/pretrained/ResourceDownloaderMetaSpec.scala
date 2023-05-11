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

import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Version
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.BeforeAndAfter

class ResourceDownloaderMetaSpec extends AnyFlatSpec with BeforeAndAfter {

  val resourcePath = "src/test/resources/resource-downloader/test_metadata.json"
  val mockResourceDownloader: MockResourceDownloader = new MockResourceDownloader(resourcePath)

  val realPrivateDownloader: ResourceDownloader = ResourceDownloader.privateDownloader
  val realPublicDownloader: ResourceDownloader = ResourceDownloader.publicDownloader
  val realCommunityDownloader: ResourceDownloader = ResourceDownloader.communityDownloader

  before {
    ResourceDownloader.privateDownloader = mockResourceDownloader
    ResourceDownloader.publicDownloader = mockResourceDownloader
    ResourceDownloader.communityDownloader = mockResourceDownloader
  }

  after {
    ResourceDownloader.privateDownloader = realPrivateDownloader
    ResourceDownloader.publicDownloader = realPublicDownloader
    ResourceDownloader.communityDownloader = realCommunityDownloader
  }

  def captureOutput(thunk: => Unit): String = {
    val stream = new java.io.ByteArrayOutputStream()
    Console.withOut(stream) {
      thunk
    }
    stream.toString
  }

  def extractTableContent(string: String): Array[String] = {
    val split = string.split("\n")
    split.slice(3, split.length - 1)
  }

  "ResourceDownloader" should "list pretrained models for an annotator using the listPretrainedResources" in {
    //    annotator, lang, version
    var resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      annotator = Some("TestAnnotator"),
      lang = Some("en"),
      Some(Version.parse("2.5.1")))
    assert(resources.length == 1)
    assert(resources.head.split(":")(0).equals("anno1"))

    //    lang, version
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      None,
      lang = Some("tst"),
      Some(Version.parse("2.5.1")))
    assert(resources.length == 1)
    assert(resources.head.split(":")(0).equals("anno_missing"))

    //    annotator, version
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      annotator = Some("TestAnnotator"),
      None,
      Some(Version.parse("2.5.1")))
    assert(resources.length == 1)
    assert(resources.head.split(":")(0).equals("anno1"))

    //    annotator, lang
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      annotator = Some("AlbertEmbeddings"),
      lang = Some("en"))
    assert(resources.length == 2)
    assert(resources.forall(_.startsWith("albert")))
    assert(resources.head.split(":")(1).equals("en"))

    //    annotator
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      annotator = Some("TestAnnotator"))
    assert(resources.length == 3)
    assert(resources.forall(_.startsWith("anno")))

    //    lang
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      None,
      lang = Some("de"))
    assert(resources.length == 2)
    assert(resources.head.split(":")(1).equals("de"))

    //    version
    resources = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      None,
      None,
      Some(Version.parse("2.4.0")))
    assert(resources.length == 3)
  }

  it should "have various interfaces for showPublicModels" in {
    val allModels = extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels()
    })
    assert(
      allModels.length == mockResourceDownloader.resources.count(
        _.category
          .getOrElse(ResourceType.NOT_DEFINED.toString)
          .equals(ResourceType.MODEL.toString)))

    val allContextSpell = extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("TestAnnotator")
    })
    assert(allContextSpell.length == 3)
    assert(allContextSpell.forall(_.contains("anno")))

    val itContextSpell = extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("TestAnnotator", "de")
    })
    assert(itContextSpell.length == 1)

    val enContextSpell = extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("TestAnnotator", "en", "2.5.1")
    })
    assert(enContextSpell.length == 1)

  }

  it should "list all available annotators" taggedAs FastTest in {
    val stream = captureOutput {
      ResourceDownloader.showAvailableAnnotators()
    }
    val allAnnotators: Set[String] =
      mockResourceDownloader.resources.map(_.annotator.getOrElse("")).toSet.filter { a =>
        !a.equals("")
      }
    val annotatorsInOutput: Set[String] = stream.split("\n").toSet
    assert(allAnnotators.diff(annotatorsInOutput).isEmpty)
    assert(annotatorsInOutput.diff(allAnnotators).isEmpty)
  }

  it should "still find annotators missing the annotator field" taggedAs FastTest in {
    val noAnnoField = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      lang = Some("tst") // defined in test_meta.json
    )
    assert(noAnnoField.length == 2)
    assert(noAnnoField.forall(_.contains("anno_")))

    val noAnnoFieldFilter = ResourceDownloader.listPretrainedResources(
      folder = "public/models",
      ResourceType.MODEL,
      annotator = Some("TestAnnotator"), // these have "annotator" field defined.
      lang = Some("tst") // defined in test_meta.json
    )
    assert(noAnnoFieldFilter.length == 1)
    assert(noAnnoFieldFilter.head.equals("anno_not_missing:tst:2.5.4"))

  }
  it should "should download a model and unzip file" taggedAs SlowTest in {
    ResourceDownloader.privateDownloader = realPrivateDownloader
    ResourceDownloader.publicDownloader = realPublicDownloader
    ResourceDownloader.communityDownloader = realCommunityDownloader
    ResourceDownloader.downloadModelDirectly(
      "public/models/bert_base_cased_es_3.2.2_3.0_1630999631885.zip")
  }

  it should "download a model and keep it as zip" taggedAs SlowTest in {
    ResourceDownloader.privateDownloader = realPrivateDownloader
    ResourceDownloader.publicDownloader = realPublicDownloader
    ResourceDownloader.communityDownloader = realCommunityDownloader
    ResourceDownloader.downloadModelDirectly(
      "s3://auxdata.johnsnowlabs.com/public/models/albert_base_sequence_classifier_ag_news_en_3.4.0_3.0_1639648298937.zip",
      folder = "public/models",
      unzip = false)
  }

  it should "be able to list from online metadata" in {
    ResourceDownloader.privateDownloader = realPrivateDownloader
    ResourceDownloader.publicDownloader = realPublicDownloader
    ResourceDownloader.communityDownloader = realCommunityDownloader

    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels()
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("NerDLModel")
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("NerDLModel", "en")
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicModels("NerDLModel", "en", "2.5.0")
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showAvailableAnnotators()
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicPipelines()
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicPipelines("en")
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showPublicPipelines("en", "2.5.0")
    }).nonEmpty)
    assert(extractTableContent(captureOutput {
      ResourceDownloader.showUnCategorizedResources("en")
    }).nonEmpty)
  }

}
