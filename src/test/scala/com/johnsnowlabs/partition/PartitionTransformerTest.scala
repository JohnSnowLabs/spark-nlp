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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.annotator.MarianTransformer
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.cleaners.Cleaner
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class PartitionTransformerTest extends AnyFlatSpec with SparkSessionTest {

  val wordDirectory = "src/test/resources/reader/doc"
  val emailDirectory = "src/test/resources/reader/email"
  val htmlDirectory = "src/test/resources/reader/html"

  "PartitionTransformer" should "work in a RAG pipeline" taggedAs SlowTest in {
    val partition = new PartitionTransformer()
      .setInputCols("doc")
      .setContentType("application/msword")
      .setContentPath(s"$wordDirectory/fake_table.docx")
      .setOutputCol("partition")

    val marian = MarianTransformer
      .pretrained()
      .setInputCols("partition")
      .setOutputCol("translation")
      .setMaxInputLength(30)

    val pipeline = new Pipeline()
      .setStages(Array(partition, marian))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.select("doc", "partition", "translation").show(truncate = false)

    assert(resultDf.select("partition").count() > 0)
  }

  it should "work with a Document input" taggedAs FastTest in {
    import spark.implicits._
    val testDataSet = Seq("An example with DocumentAssembler annotator").toDS.toDF("text")

    val partition = new PartitionTransformer()
      .setInputCols("document")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(testDataSet)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

  it should "work with a Cleaner input" taggedAs FastTest in {
    import spark.implicits._
    val testDf = Seq("\\x88This text contains ®non-ascii characters!●").toDS.toDF("text")

    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_non_ascii_chars")

    val partition = new PartitionTransformer()
      .setInputCols("cleaned")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, cleaner, partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(testDf)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

  it should "set headers for URLs" taggedAs SlowTest in {
    import spark.implicits._
    val testDataSet = Seq("https://www.blizzard.com", "https://www.google.com/").toDS.toDF("text")

    val partition = new PartitionTransformer()
      .setInputCols("document")
      .setOutputCol("partition")
      .setContentType("url")
      .setHeaders(Map("Accept-Language" -> "es-ES"))

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(testDataSet)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

  it should "partition Word documents" taggedAs FastTest in {
    val partition = new PartitionTransformer()
      .setInputCols("document")
      .setContentPath(s"$emailDirectory")
      .setContentType("message/rfc822")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

  it should "partition PDF documents" taggedAs FastTest in {
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")

    dummyDataFrame.show()

    val partition = new PartitionTransformer()
      .setInputCols("content")
      .setContentType("application/pdf")
      .setOutputCol("partition")
      .setStoreSplittedPdf(true)

    val pipeline = new Pipeline()
      .setStages(Array(partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(dummyDataFrame)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

  it should "partition HTML documents" taggedAs FastTest in {
    val partition = new PartitionTransformer()
      .setInputCols("text")
      .setContentPath(s"$htmlDirectory")
      .setContentType("text/html")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show()

    assert(resultDf.select("partition").count() > 0)
  }

}
