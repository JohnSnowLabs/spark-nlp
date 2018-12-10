package com.johnsnowlabs.util

import java.io.File

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher, SparkAccessor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec

import scala.collection.mutable.ArrayBuffer


trait ExportCSVToolBehaviors  { this: FlatSpec =>

  def testPOSModelBuilder(dataset: Dataset[Row]): PipelineModel = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val POSTag = PerceptronModel.pretrained()

    val finisher = new Finisher()
      .setInputCols("token","pos")
      .setIncludeMetadata(true)
      //.setOutputAsArray(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        POSTag,
        finisher
      ))

    val model = pipeline.fit(dataset)
    model
  }

  def testExportToCoNLLFile(dataset: Dataset[Row], outputFilePath: String): Unit = {
    it should "successfully generate POS tags" ignore  {
      import SparkAccessor.spark.implicits._ //for col

      dataset.show(false)
      val model = this.testPOSModelBuilder(dataset)
      val POSdataset = model.transform(dataset)
      POSdataset.show()
      POSdataset.select("finished_token_metadata").show(false)

      val newPOSDataset = POSdataset.select("finished_token", "finished_pos", "finished_token_metadata").
                                as[(Array[String], Array[String], Array[(String, String)])]
      newPOSDataset.show()

      val CoNLLDataset = newPOSDataset.flatMap(row => {
        val newColumns: ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
        val columns = (row._1 zip row._2 zip row._3.map(_._2.toInt)).map{case (a,b) => (a._1, a._2, b)}
        var sentenceId = 1
        columns.foreach(a => {
          if (a._3 != sentenceId){
            newColumns.append(("", "", "", ""))
            sentenceId = a._3
          }
          newColumns.append((a._1, a._2, a._2, "O"))
        })
        newColumns
      })

      CoNLLDataset.show()

      CoNLLDataset.coalesce(1).write.format("com.databricks.spark.csv").
                option("delimiter", " ").
                save(outputFilePath)
    }
  }

  def getListOfFiles(filesPath: String):List[File] = {
    val directory = new File(filesPath)
    if (directory.exists && directory.isDirectory) {
      directory.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def testExportSeveralCoNLLFiles(filesPath: String): Unit = {
    it should "successfully generate POS tags" ignore {
      import SparkAccessor.spark.implicits._ //for toDS and toDF

      val listOfFiles = getListOfFiles(filesPath)

      listOfFiles.foreach { filePath =>

        val data = SparkAccessor.spark.sparkContext.wholeTextFiles(filePath.toString).toDS.toDF("filename", "text")

        val model = this.testPOSModelBuilder(data)
        val POSdataset = model.transform(data)

        val newPOSDataset = POSdataset.select("finished_token", "finished_pos", "finished_token_metadata").
          as[(Array[String], Array[String], Array[(String, String)])]

        val CoNLLDataset = newPOSDataset.flatMap(row => {
          val newColumns: ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
          val columns = (row._1 zip row._2 zip row._3.map(_._2.toInt)).map{case (a,b) => (a._1, a._2, b)}
          var sentenceId = 1
          columns.foreach(a => {
            if (a._3 != sentenceId){
              newColumns.append(("", "", "", ""))
              sentenceId = a._3
            }
            newColumns.append((a._1, a._2, a._2, "O"))
          })
          newColumns
        })
        val CSVFilePath = filePath.toString.replaceAll(".txt", ".csv")
        CoNLLDataset.coalesce(1).write.format("com.databricks.spark.csv").
            option("delimiter", " ").
            save(CSVFilePath)
      }

    }
  }

  def testEvaluation(dataset: Dataset[Row]): Unit = {
    it should "successfully generate POS tags" ignore {

      import SparkAccessor.spark.implicits._ //for .as

      dataset.show()

      val newPOSDataset = dataset.select("token", "pos", "result", "metadata").
        as[(Array[String], Array[String],  Array[String], Array[(String, String)])]
      newPOSDataset.show()

      val CoNLLDataset = newPOSDataset.flatMap(row => {
        val newColumns: ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
        val columns = (row._1 zip row._2 zip row._3 zip row._4.map(_._2.toInt)).map{case (a, b) =>
          (a._1._1, a._1._2, a._2, b) }
        var sentenceId = 1
        columns.foreach(a => {
          if (a._4 != sentenceId){
            newColumns.append(("", "", "", ""))
            sentenceId = a._4
          }
          newColumns.append((a._1, a._2, a._2, a._3))
        })
        newColumns
      })

      CoNLLDataset.show()

      CoNLLDataset.coalesce(1).write.format("com.databricks.spark.csv").
        option("delimiter", " ").
        save("evaluation/evaluation.csv")
    }
  }

}
