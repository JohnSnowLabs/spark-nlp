package com.johnsnowlabs.nlp.annotators.sentence_detector_dl


import com.johnsnowlabs.nlp.SparkAccessor.spark

import scala.io.Source
import com.johnsnowlabs.ml.tensorflow.{TensorflowGenericClassifier, TensorflowWrapper, Variables}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.io.IOUtils
import org.scalatest.FlatSpec
import org.tensorflow.Graph
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import org.apache.spark.ml.Pipeline

import scala.util.Random

class SentenceDetectorDLSpec  extends FlatSpec {
  implicit val session = spark

  val dataset = "en"//europarl

  val trainDataFile = s"/data/sent/lang/${dataset}.train.txt"
  val testDataFile = s"/data/sent/lang/${dataset}.test.txt"
  val testSampleFreeText = "/data/sent/sample.text"

  val vocabFile = s"/data/sent/vocab_${dataset}.json"
  val modelGraphFile = "/models/sent/graphs/cnn.pb"
  val savedTFModelPath = s"/models/sent/cnn_${dataset}"
  val savedModelPath = s"/models/SDDL_${dataset}_cnn"

  val languages = Array("bg", "bs", "de", "el", "en", "hr", "mk", "ro", "sq", "sr", "tr", "multilang")

  "Sentence Detector DL" should "train a new model" in {

    languages.foreach(lang => {

      println(s"Training ${lang}")

      val trainDataFile = s"/data/sent/lang/${lang}.train.txt"
      val savedModelPath = s"/models/SDDL_${lang}_cnn"

      assert(Files.exists(Paths.get(trainDataFile)))

      val df = spark.read.text(trainDataFile).toDF("text")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val sentenceDetector = new SentenceDetectorDLApproach()
        .setInputCols(Array("document"))
        .setOutputCol("sentences")
        .setGraphFile("/models/sent/graphs/cnn_large_vocabulary.pb")
        .setEpochsNumber(5)

      val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))

      val model = pipeline.fit(df)
      model.stages(1).asInstanceOf[SentenceDetectorDLModel].write.overwrite().save(savedModelPath)

      println(s"${lang} completed.\n\n")
    })
  }

  "Sentence Detector DL" should "run test metrics" in {

    assert(Files.exists(Paths.get(testDataFile)))
    assert(Files.exists(Paths.get(savedModelPath)))

    var begin = true
    languages.foreach(lang => {

      val testDataFile = s"/data/sent/lang/${lang}.test.txt"

      val text = Source.fromFile(testDataFile).getLines().map(_.trim).mkString("\n")

      val savedModelPath = s"/models/SDDL_${lang}_cnn"

      val sentenceDetectorDL = SentenceDetectorDLModel.load(savedModelPath)
      val metrics = sentenceDetectorDL.getMetrics(text, false)

      if (begin) {
        println("%5$15s %1$15s %2$15s %3$15s %4$15s".format("Accuracy", "Recall", "Precision", "F1", "Language"))
        begin = false
      }
      println("%5$15s %1$15.2f %2$15.2f %3$15.2f %4$15.2f".format(
        metrics.accuracy,
        metrics.recall,
        metrics.precision,
        metrics.f1,
        lang))

    })
  }

  "Sentence Detector DL" should "load and run pretrained model" in {
    assert(Files.exists(Paths.get(testSampleFreeText)))
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
          println(text.slice(s.begin, s.end))
          println("\n")
        })
      }

    })
  }

  "Sentence Detector DL" should "save new pretrained model" ignore {

    val sentenceDetectorDL = new SentenceDetectorDLModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")
      .setModel("cnn")
      .setupNew(session, savedTFModelPath, vocabFile)

    sentenceDetectorDL.write.overwrite().save(savedModelPath)
  }

  "Sentence Detector DL" should "train using GenericClassifier" ignore {

    assert(Files.exists(Paths.get(trainDataFile)))

    val text = Source.fromFile(trainDataFile).getLines().map(_.trim).mkString("\n")

    val encoder = new SentenceDetectorDLEncoder()
    if (Files.exists(Paths.get(vocabFile))){
      encoder.loadVocabulary(vocabFile)
    } else {
      encoder.buildVocabulary(text)
      encoder.saveVocabulary(vocabFile)
    }

    val data = encoder.getTrainingData(text)

    println("Positive examples %s".format(data._1.filter(l => l == 1.0f).length))
    println("Negative examples %s".format(data._1.filter(l => l == 0.0f).length))


    val graph = new Graph()
    val graphStream = ResourceHelper.getResourceStream(modelGraphFile)
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(graphBytesDef)

    val tfWrapper = new TensorflowWrapper(
      Variables(Array.empty[Byte], Array.empty[Byte]),
      graph.toGraphDef
    )

    val tfModel = new TensorflowGenericClassifier(tfWrapper, outputLogsPath = None)

    tfModel.train(
      data._2,
      data._1.map(Array(_)),
      batchSize = 32,
      epochsNumber = 20,
      learningRate = 0.0001f,
      validationSplit = 0.1f,
      classWeights = Array(1.0f),
      dropout = 1.0f,
      uuid = "sentence_detector_dl"
    )

    tfModel.model.saveToFile(savedTFModelPath)
  }



}
