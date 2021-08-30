/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.sentence_detector_dl


import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.tags.FastTest

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source
import java.nio.file.{Files, Paths}

class SentenceDetectorDLSpec extends AnyFlatSpec {
  implicit val session: SparkSession = spark

  val trainDataFile = "src/test/resources/sentence_detector_dl/train.txt"
  val testDataFile = "src/test/resources/sentence_detector_dl/test.txt"
  val savedModelPath = "./tmp_sdd_model/new_model"
  val testSampleFreeText = "src/test/resources/sentence_detector_dl/sample.txt"

  import spark.implicits._

  "Sentence Detector DL" should "train a new model" taggedAs FastTest in {

    assert(Files.exists(Paths.get(trainDataFile)))

    val df = spark.read.text(trainDataFile).toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorDLApproach()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")
      .setEpochsNumber(100)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))

    val model = pipeline.fit(df)
    model.stages(1).asInstanceOf[SentenceDetectorDLModel].write.overwrite().save(savedModelPath)
  }

  "Sentence Detector DL" should "run test metrics" taggedAs FastTest in {

    val text = Source.fromFile(testDataFile).getLines().map(_.trim).mkString("\n")
    val sentenceDetectorDL = SentenceDetectorDLModel.load(savedModelPath)
    val metrics = sentenceDetectorDL.getMetrics(text, false)


    println("%1$15s %2$15s %3$15s %4$15s".format("Accuracy", "Recall", "Precision", "F1"))

    println("%1$15.2f %2$15.2f %3$15.2f %4$15.2f".format(
      metrics.accuracy,
      metrics.recall,
      metrics.precision,
      metrics.f1))
  }

  "Sentence Detector DL" should "load and run pretrained model" taggedAs FastTest in {

    val text = Source.fromFile(testSampleFreeText).getLines().mkString("\n")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel
      .load(savedModelPath)
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetectorDL))

    case class textOnly(text: String)


    val emptyDataset = spark.emptyDataset[String]

    val lightModel = new LightPipeline(pipeline.fit(emptyDataset))

    lightModel.fullAnnotate(text).foreach(anno => {
      if (anno._1 == "sentences") {
        anno._2.foreach(s => {
          //          println(s.result)
          //          println("\n")
        })
      }

    })
  }

  "Sentence Detector DL" should "download and run pretrained model" taggedAs FastTest in {

    val text = Source.fromFile(testSampleFreeText).getLines().mkString("\n")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel.pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetectorDL))

    case class textOnly(text: String)

    import spark.implicits._

    val emptyDataset = spark.emptyDataset[String]

    val lightModel = new LightPipeline(pipeline.fit(emptyDataset))

    lightModel.fullAnnotate(text).foreach(anno => {
      if (anno._1 == "sentences") {
        anno._2.foreach(s => {
          //          println(s.result)
          //          println("\n")
        })
      }

    })
  }

}
