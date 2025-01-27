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
package com.johnsnowlabs.nlp.annotators.cleaners

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class ExtractorTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  val emlData =
    "from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by\n  \\n ABC.DEF.local2 ([ba23::58b5:2236:45g2:88h2%25]) with mapi id\\\n  n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200"

  "Extractor" should "be able to extract dates" in {
    val dateExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("date")
      .setExtractorMode("email_date")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, dateExtractor))

    val testDf = Seq(
      emlData,
      "First date Fri, 26 Mar 2021 11:04:09 +1200 and then another date Wed, 26 Jul 2025 11:04:09 +1200").toDS
      .toDF("text")
    testDf.show(truncate = false)
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("date").show(truncate = false)
  }

  it should "be able to extract email addresses" in {
    val emailExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("email")
      .setExtractorMode("email_address")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, emailExtractor))

    val testDf = Seq(
      "Me me@email.com and You <You@email.com>\n  ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)").toDS
      .toDF("text")
    testDf.show(truncate = false)
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("email").show(truncate = false)
  }

  it should "be able to extract IPv4 and IPv6 addresses" in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))

    val testDf = Seq(
      "Me me@email.com and You <You@email.com> ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)").toDS
      .toDF("text")
    testDf.show(truncate = false)
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("ip").show(truncate = false)
  }

  it should "be able to extract only IPv4 addresses" in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address")
      .setIPAddressPattern(
        "(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)){3}")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))

    val testDf = Seq(
      "Me me@email.com and You <You@email.com> ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)").toDS
      .toDF("text")
    testDf.show(truncate = false)
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("ip").show(truncate = false)
  }

  it should "be able to extract only IP address name" in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address_name")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))
    val testDf = Seq(emlData).toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("ip").show(truncate = false)
  }

  it should "be able to extract only MAPI IDs" in {
    val mapiIdExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("mapi_id")
      .setExtractorMode("mapi_id")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, mapiIdExtractor))
    val testDf = Seq(emlData).toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("mapi_id").show(truncate = false)
  }

  it should "be able to extract US phone numbers" in {
    val usPhonesExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("us_phone")
      .setExtractorMode("us_phone_numbers")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, usPhonesExtractor))
    val testDf = Seq("Phone number: 215-867-5309").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("us_phone").show(truncate = false)
  }

  it should "be able to extract bullets" in {
    val bulletExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("bullets")
      .setExtractorMode("bullets")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bulletExtractor))
    val testDf = Seq(
      "1.1 This is a very important point",
      "a.1 This is a very important point",
      "1.4.2 This is a very important point",
      "• bullet 1").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("bullets").show(truncate = false)
  }

  it should "be able to extract image URLs" in {
    val imageUrlExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("image_urls")
      .setExtractorMode("image_urls")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, imageUrlExtractor))
    val testDf = Seq("""
      <img src="https://example.com/images/photo1.jpg" />
      <img src="https://example.org/assets/icon.png" />
      <link href="https://example.net/style.css" />
      """).toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("image_urls").show(truncate = false)
  }

  it should "be able to extract text after" in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_after")
      .setExtractorMode("text_after")
      .setTextPattern("SPEAKER \\d{1}:")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("SPEAKER 1: Look at me, I'm flying!").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("text_after").show(truncate = false)
  }

  it should "be able to extract text before" in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_before")
      .setExtractorMode("text_before")
      .setTextPattern("STOP")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("Here I am! STOP Look at me! STOP I'm flying! STOP").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    resultDf.select("text_before").show(truncate = false)
  }

}
