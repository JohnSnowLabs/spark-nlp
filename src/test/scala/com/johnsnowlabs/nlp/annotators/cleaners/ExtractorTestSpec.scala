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

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class ExtractorTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  val emlData =
    "from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by\n  \\n ABC.DEF.local2 ([ba23::58b5:2236:45g2:88h2%25]) with mapi id\\\n  n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200"

  "Extractor" should "be able to extract dates" taggedAs FastTest in {
    val dateExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("date")
      .setExtractorMode("email_date")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, dateExtractor))
    val testDf = Seq(
      emlData,
      "First date Fri, 26 Mar 2021 11:04:09 +1200 and then another date Wed, 26 Jul 2025 11:04:09 +1200").toDS
      .toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "date")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(
      Seq("Fri, 26 Mar 2021 11:04:09 +1200"),
      Seq("Fri, 26 Mar 2021 11:04:09 +1200", "Wed, 26 Jul 2025 11:04:09 +1200"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract email addresses" taggedAs FastTest in {
    val emailExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("email")
      .setExtractorMode("email_address")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, emailExtractor))
    val testDf = Seq(
      "Me me@email.com and You <You@email.com>\n  ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)",
      "Im Rabn <Im.Rabn@npf.gov.nr>").toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "email")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("me@email.com", "You@email.com"), Seq("Im.Rabn@npf.gov.nr"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract IPv4 and IPv6 addresses" taggedAs FastTest in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))
    val testDf = Seq("""from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by
    \n ABC.DEF.local ([68.183.71.12]) with mapi id\
    n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200""").toDS
      .toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "ip")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("ba23::58b5:2236:45g2:88h2", "68.183.71.12"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract only IPv4 addresses" taggedAs FastTest in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address")
      .setIpAddressPattern(
        "(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)){3}")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))
    val testDf =
      Seq("Me me@email.com and You <You@email.com> ([ba23::58b5:2236:45g2:88h2]) (10.0.2.0)").toDS
        .toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "ip")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("10.0.2.0"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract only IP address name" taggedAs FastTest in {
    val ipAddressExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("ip")
      .setExtractorMode("ip_address_name")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, ipAddressExtractor))
    val testDf = Seq(emlData).toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "ip")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("ABC.DEF.local", "ABC.DEF.local"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract only MAPI IDs" taggedAs FastTest in {
    val mapiIdExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("mapi_id")
      .setExtractorMode("mapi_id")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, mapiIdExtractor))
    val testDf = Seq(emlData).toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "mapi_id")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("32.88.5467.123"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract US phone numbers" taggedAs FastTest in {
    val usPhonesExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("us_phone")
      .setExtractorMode("us_phone_numbers")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, usPhonesExtractor))
    val testDf =
      Seq("215-867-5309", "Phone Number: +1 215.867.5309", "Phone Number: Just Kidding").toDS
        .toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "us_phone")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("215-867-5309"), Seq("+1 215.867.5309"), Seq())

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract bullets" taggedAs FastTest in {
    val bulletExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("bullets")
      .setExtractorMode("bullets")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bulletExtractor))
    val testDf = Seq(
      "1. Introduction:",
      "a. Introduction:",
      "20.3 Morse code ●●●",
      "5.3.1 Convolutional Networks",
      "D.b.C Recurrent Neural Networks",
      "2.b.1 Recurrent Neural Networks",
      "eins. Neural Networks",
      "bb.c Feed Forward Neural Networks",
      "aaa.ccc Metrics",
      "version = 3.8",
      "1 2. 3 4",
      "1) 2. 3 4",
      "2",
      "1..2.3 four",
      "Fig. 2: The relationship",
      "23 is everywhere",
      "• bullet 1").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "bullets")
    val actualResult: Array[Seq[String]] = resultAnnotation.map(_.map(_.result))
    val expectedResult: Array[Seq[String]] = Array(
      Seq("(1,None,None)"),
      Seq("(a,None,None)"),
      Seq("(20,3,None)"),
      Seq("(5,3,1)"),
      Seq("(D,b,C)"),
      Seq("(2,b,1)"),
      Seq("(None,None,None)"),
      Seq("(bb,c,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"),
      Seq("(None,None,None)"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract image URLs" taggedAs FastTest in {
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

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "image_urls")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult =
      Array(Seq("https://example.com/images/photo1.jpg", "https://example.org/assets/icon.png"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract images for different cases" taggedAs FastTest in {
    val imageUrlExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("image_urls")
      .setExtractorMode("image_urls")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, imageUrlExtractor))
    val testDf = Seq(
      "https://my-image.jpg",
      "https://my-image.png with some text",
      "https://my-image/with/some/path.png",
      "some text https://my-image.jpg with another http://my-image.bmp",
      "http://not-an-image.com",
      "some text",
      "some text https://my-image.JPG with ano100" +
        "ther http://my-image.BMP",
      "http://my-path-with-CAPS/my-image.JPG",
      "http://my-path/my%20image.JPG",
      "https://my-image.jpg#ref").toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "image_urls")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(
      Seq("https://my-image.jpg"),
      Seq("https://my-image.png"),
      Seq("https://my-image/with/some/path.png"),
      Seq("https://my-image.jpg", "http://my-image.bmp"),
      Seq(),
      Seq(),
      Seq("https://my-image.JPG", "http://my-image.BMP"),
      Seq("http://my-path-with-CAPS/my-image.JPG"),
      Seq("http://my-path/my%20image.JPG"),
      Seq("https://my-image.jpg"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract text after" taggedAs FastTest in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_after")
      .setExtractorMode("text_after")
      .setTextPattern("SPEAKER \\d{1}:")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("SPEAKER 1: Look at me, I'm flying!").toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "text_after")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("Look at me, I'm flying!"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract text after with a pattern with punctuation" taggedAs FastTest in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_after")
      .setExtractorMode("text_after")
      .setTextPattern("BLAH;")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("Teacher: BLAH BLAH BLAH; Student: BLAH BLAH BLAH!").toDS.toDF("text")
    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "text_after")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("Student: BLAH BLAH BLAH!"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract text before" taggedAs FastTest in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_before")
      .setExtractorMode("text_before")
      .setTextPattern("STOP")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("Here I am! STOP Look at me! STOP I'm flying! STOP").toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "text_before")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("Here I am!"))

    actualResult shouldEqual expectedResult
  }

  it should "be able to extract text before with index" taggedAs FastTest in {
    val textAfterExtractor = new Extractor()
      .setInputCols("document")
      .setOutputCol("text_before")
      .setExtractorMode("text_before")
      .setTextPattern("BLAH")
      .setIndex(1)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, textAfterExtractor))
    val testDf = Seq("Teacher: BLAH BLAH BLAH; Student: BLAH BLAH BLAH!").toDS.toDF("text")

    val resultDf = pipeline.fit(testDf).transform(testDf)

    val resultAnnotation = AssertAnnotations.getActualResult(resultDf, "text_before")
    val actualResult = resultAnnotation.map(_.map(_.result))
    val expectedResult = Array(Seq("Teacher: BLAH"))

    actualResult shouldEqual expectedResult
  }

}
