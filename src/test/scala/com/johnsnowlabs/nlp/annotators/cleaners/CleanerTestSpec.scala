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
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class CleanerTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "Cleaner" should "convert an output string that looks like a byte string to a string using the specified encoding" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("bytes_string_to_string")

    val testDf =
      Seq("This is a test with regular text", "Hello ð\\x9f\\x98\\x80").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  "Cleaner" should "clean text" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean")
      .setBullets(true)
      .setExtraWhitespace(true)
      .setDashes(true)

    val testDf = Seq("● An excellent point!", "ITEM 1A:     RISK-FACTORS").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  "Cleaner" should "clean non-ascii characters" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_non_ascii_chars")

    val testDf = Seq("\\x88This text contains ®non-ascii characters!●").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  "Cleaner" should "clean ordered bullets" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_ordered_bullets")

    val testDf = Seq(
      "1.1 This is a very important point",
      "a.1 This is a very important point",
      "1.4.2 This is a very important point").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  it should "clean postfix" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_postfix")
      .setCleanPrefixPattern("(END|STOP)")

    val testDf = Seq("The end! END").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  it should "clean prefix" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_prefix")
      .setCleanPrefixPattern("(SUMMARY|DESCRIPTION):")

    val testDf = Seq("SUMMARY: This is the best summary of all time!").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  it should "remove punctuation" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("remove_punctuation")

    val testDf = Seq("$A lovely quote!”").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  it should "replace unicode quotes" taggedAs FastTest in {
    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("replace_unicode_characters")

    val testDf = Seq(
      """\x93A lovely quote!\x94""",
      """\x91A lovely quote!\x92""",
      """"\u201CA lovely quote!\u201D — with a dash"""").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

  it should "translate text" taggedAs SlowTest in {
    val cleaner = Cleaner
      .pretrained()
      .setInputCols("document")
      .setOutputCol("cleaned")

    val testDf = Seq("This should go to French").toDS.toDF("text")
    testDf.show(truncate = false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, cleaner))

    val resultDf = pipeline.fit(testDf).transform(testDf)
    resultDf.select("cleaned").show(truncate = false)
  }

}
