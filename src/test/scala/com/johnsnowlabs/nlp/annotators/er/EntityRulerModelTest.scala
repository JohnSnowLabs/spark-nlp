package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.er.EntityRulerFixture.{email1, email2, ip1, ip2}
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerModelTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  private val trainedModel: EntityRulerModel = getTrainedModel
  private val basePipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, trainedModel))
  private val sampleData = Seq(
    s"Contact $email1 or $email2 from $ip1 or $ip2."
  ).toDF("text")

  "EntityRulerModel (default)" should "extract both email and IP entities" in {
    trainedModel.setExtractEntities(Array())

    val resultDf = basePipeline.fit(sampleData).transform(sampleData)

    val result = AssertAnnotations.getActualResult(resultDf, "entities").flatten
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

    val result = AssertAnnotations.getActualResult(resultDf, "entities").flatten
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

    val result = AssertAnnotations.getActualResult(resultDf, "entities").flatten
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

    val result = AssertAnnotations.getActualResult(resultDf, "entities").flatten
      .map(annotation => annotation.result)
    assert(result.length == 4)
    assert(result.contains(email1))
    assert(result.contains(email2))
    assert(result.contains(ip1))
    assert(result.contains(ip2))
  }

  private def getTrainedModel: EntityRulerModel = {
    val entityRulerApproach = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        path = "src/test/resources/entity-ruler/email-ip-regex.json",
        readAs = ReadAs.TEXT,
        options = Map("format" -> "JSON")
      )
      .setSentenceMatch(false)
      .setUseStorage(false)

    entityRulerApproach.fit(emptyDataSet)
  }

}
