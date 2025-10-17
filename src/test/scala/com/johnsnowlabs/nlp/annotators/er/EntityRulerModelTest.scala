/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.er.EntityRulerFixture.{email1, email2, ip1, ip2}
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerModelTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  private val trainedModel: EntityRulerModel = getTrainedModel
  private val basePipeline =
    new Pipeline().setStages(Array(documentAssembler, tokenizer, trainedModel))
  private val sampleData = Seq(s"Contact $email1 or $email2 from $ip1 or $ip2.").toDF("text")

  "EntityRulerModel (default)" should "extract both email and IP entities" in {
    trainedModel.setExtractEntities(Array())

    val resultDf = basePipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)
    assert(result.length == 4)
    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  "EntityRulerModel (email-only)" should "extract only email addresses" in {
    trainedModel.setExtractEntities(Array("EMAIL"))

    val resultDf = basePipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)
    assert(result.length == 2)
    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(!result.contains(ip1))
    assert(!result.contains(ip2))
  }

  "EntityRulerModel (ip-only)" should "extract only IP addresses" in {
    trainedModel.setExtractEntities(Array("IP_ADDRESS"))

    val resultDf = basePipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)
    assert(result.length == 2)
    assert(!result.contains(email1))
    assert(!result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  "EntityRulerModel (email+ip)" should "extract both emails and IPs when both requested" in {
    trainedModel.setExtractEntities(Array("EMAIL", "IP_ADDRESS"))

    val resultDf = basePipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)
    assert(result.length == 4)
    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  "EntityRulerModel (autoMode='contact_entities')" should "extract emails" in {
    val autoModel = new EntityRulerModel()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setAutoMode("contact_entities") // activates EMAIL + IP patterns internally

    val autoPipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, autoModel))

    val resultDf = autoPipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)

    assert(result.length == 4)
    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  it should "detect only emails and phones with COMMUNICATION_ENTITIES mode" in {
    val resultDf = runPipeline(EntityRulerModel.COMMUNICATION_ENTITIES)
    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)

    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.length == 2)
  }

  it should "detect only IPs and hostnames with NETWORK_ENTITIES mode" in {
    val resultDf = runPipeline(EntityRulerModel.NETWORK_ENTITIES)
    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)

    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  it should "detect both emails, phones, and IPs with CONTACT_ENTITIES mode" in {
    val resultDf = runPipeline(EntityRulerModel.CONTACT_ENTITIES)
    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)

    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
    assert(result.length >= 4)
  }

  it should "detect everything with ALL_ENTITIES mode" in {
    val resultDf = runPipeline(EntityRulerModel.ALL_ENTITIES)
    val result = AssertAnnotations
      .getActualResult(resultDf, "entities")
      .flatten
      .map(annotation => annotation.result)

    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
    assert(result.length >= 4)
  }

  it should "handle case-insensitive autoMode values" in {
    val mixedCaseModel = runPipeline("Contact_Entities")
    val upperCaseModel = runPipeline("CONTACT_ENTITIES")
    val lowerCaseModel = runPipeline("contact_entities")

    val e1 = mixedCaseModel.selectExpr("explode(entities.result)").as[String].collect().sorted
    val e2 = upperCaseModel.selectExpr("explode(entities.result)").as[String].collect().sorted
    val e3 = lowerCaseModel.selectExpr("explode(entities.result)").as[String].collect().sorted

    assert(e1.sameElements(e2))
    assert(e2.sameElements(e3))
  }

  it should "detect email addresses using sentence-level regex matching" in {
    val data =
      Seq("Please contact us at info@mydomain.com or support@helpdesk.net for assistance.").toDF(
        "text")

    val model = new EntityRulerModel()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setAutoMode(EntityRulerModel.COMMUNICATION_ENTITIES)
      .setSentenceMatch(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, model))
    val result = pipeline.fit(data).transform(data)
    val entities = result.selectExpr("explode(entities.result)").as[String].collect()

    assert(entities.contains("info@mydomain.com"))
    assert(entities.contains("support@helpdesk.net"))
    assert(entities.length == 2)
  }

  it should "detect full IP range patterns at sentence-level when using NETWORK_ENTITIES mode" in {
    val data =
      Seq("This server list includes 192.168.1.1, 10.0.0.45 and 172.16.0.2 for internal routing.")
        .toDF("text")

    val model = new EntityRulerModel()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setAutoMode(EntityRulerModel.NETWORK_ENTITIES)
      .setSentenceMatch(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, model))
    val result = pipeline.fit(data).transform(data)

    val entities = result.selectExpr("explode(entities.result)").as[String].collect()

    // We expect all three IPs to match since sentence regex scans full content
    assert(entities.exists(_.contains("192.168.1.1")))
    assert(entities.exists(_.contains("10.0.0.45")))
    assert(entities.exists(_.contains("172.16.0.2")))
    assert(entities.length == 3)
  }

  it should "handle case-insensitive autoMode with sentenceMatch = true" in {
    val data = Seq("Reach us via contact@team.io or sales@biz.org — thank you.").toDF("text")

    val model = new EntityRulerModel()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setAutoMode("contact_entities") // lowercase intentional
      .setSentenceMatch(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, model))
    val result = pipeline.fit(data).transform(data)

    val entities = result.selectExpr("explode(entities.result)").as[String].collect()

    assert(entities.exists(_.contains("contact@team.io")))
    assert(entities.exists(_.contains("sales@biz.org")))
    assert(entities.length == 2)
  }

  private def getTrainedModel: EntityRulerModel = {
    val entityRulerApproach = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        path = "src/test/resources/entity-ruler/email-ip-regex.json",
        readAs = ReadAs.TEXT,
        options = Map("format" -> "JSON"))
      .setSentenceMatch(false)
      .setUseStorage(false)

    entityRulerApproach.fit(emptyDataSet)
  }

  private def runPipeline(autoMode: String): DataFrame = {
    val model = new EntityRulerModel()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setAutoMode(autoMode)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, model))
    pipeline.fit(sampleData).transform(sampleData)
  }

}
