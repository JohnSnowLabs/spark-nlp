package com.johnsnowlabs.nlp.annotators.sentence_detector_dl


import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

import java.nio.file.{Files, Paths}
import scala.io.Source

class SentenceDetectorDLSpec  extends FlatSpec {
  implicit val session = spark

  val trainDataFile = "src/test/resources/sentence_detector_dl/train.txt"
  val testDataFile =  "src/test/resources/sentence_detector_dl/test.txt"
  val savedModelPath = "./tmp_sdd_model/new_model"
  val testSampleFreeText = "src/test/resources/sentence_detector_dl/sample.txt"

  "Sentence Detector DL" should "train a new model" taggedAs SlowTest in {

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

  "Sentence Detector DL" should "load and run pretrained model" taggedAs SlowTest in {

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

    import spark.implicits._

    val emptyDataset = spark.emptyDataset[String]

    val lightModel = new LightPipeline(pipeline.fit(emptyDataset))

    lightModel.fullAnnotate(text).foreach(anno => {
      if (anno._1 == "sentences"){
        anno._2.foreach(s => {
          println(s.result)
          println("\n")
        })
      }

    })
  }

  "Sentence Detector DL" should "download and run pretrained model" taggedAs SlowTest in {

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
      if (anno._1 == "sentences"){
        anno._2.foreach(s => {
          println(s.result)
          println("\n")
        })
      }

    })
  }

}
