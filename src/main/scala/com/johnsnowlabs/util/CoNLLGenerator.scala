/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.util

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import org.apache.commons.text.StringEscapeUtils.escapeJava
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object CoNLLGenerator {

  def exportConllFiles(
      spark: SparkSession,
      filesPath: String,
      pipelineModel: PipelineModel,
      outputPath: String): Unit = {
    import spark.implicits._ // for toDS and toDF
    val data = spark.sparkContext.wholeTextFiles(filesPath).toDS.toDF("filename", "text")
    exportConllFiles(data, pipelineModel, outputPath)
  }

  def exportConllFiles(
      spark: SparkSession,
      filesPath: String,
      pipelinePath: String,
      outputPath: String): Unit = {
    val model = PipelineModel.load(pipelinePath)
    exportConllFiles(spark, filesPath, model, outputPath)
  }

  def exportConllFiles(
      data: DataFrame,
      pipelineModel: PipelineModel,
      outputPath: String): Unit = {
    val POSdataset = pipelineModel.transform(data)
    exportConllFilesFromField(POSdataset, outputPath, "sentence")
  }

  def exportConllFiles(data: DataFrame, pipelinePath: String, outputPath: String): Unit = {
    val model = PipelineModel.load(pipelinePath)
    exportConllFiles(data, model, outputPath)
  }

  def exportConllFiles(data: DataFrame, outputPath: String): Unit = {
    exportConllFilesFromField(data, outputPath, "sentence")
  }

  def exportConllFilesFromField(
      data: DataFrame,
      outputPath: String,
      metadataSentenceKey: String): Unit = {
    import data.sparkSession.implicits._ // for udf
    var dfWithNER = data
    // if data does not contain ner column, add "O" as default
    if (Try(data("finished_ner")).isFailure) {
      def OArray = (len: Int) => { // create array of $len "O"s
        var z = new Array[String](len)
        for (i <- 0 until z.length) {
          z(i) = "O"
        }
        z
      }

      val makeOArray = data.sparkSession.udf.register("finished_pos", OArray)
      dfWithNER = data.withColumn("finished_ner", makeOArray(size(col("finished_pos"))))
    }

    val newPOSDataset = dfWithNER
      .select("finished_token", "finished_pos", "finished_token_metadata", "finished_ner")
      .as[(Array[String], Array[String], Array[(String, String)], Array[String])]
    val CoNLLDataset = makeConLLFormat(newPOSDataset, metadataSentenceKey)
    CoNLLDataset
      .coalesce(1)
      .write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .options(scala.collection.Map("delimiter" -> " ", "emptyValue" -> ""))
      .save(outputPath)
  }

  def makeConLLFormat(
      newPOSDataset: Dataset[
        (Array[String], Array[String], Array[(String, String)], Array[String])],
      metadataSentenceKey: String = "sentence"): Dataset[(String, String, String, String)] = {
    import newPOSDataset.sparkSession.implicits._ // for row casting
    newPOSDataset.flatMap(row => {
      val newColumns: ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
      val columns =
        (
          (row._1 zip row._2),
          row._3.filter(_._1 == metadataSentenceKey).map(_._2.toInt),
          row._4).zipped
          .map { case (a, b, c) =>
            (a._1, a._2, b, c)
          }
      var sentenceId = 1
      newColumns.append(("", "", "", ""))
      newColumns.append(("-DOCSTART-", "-X-", "-X-", "O"))
      newColumns.append(("", "", "", ""))
      columns.foreach(a => {
        if (a._3 != sentenceId) {
          newColumns.append(("", "", "", ""))
          sentenceId = a._3
        }
        newColumns.append((escapeJava(a._1), a._2, a._2, a._4))
      })
      newColumns
    })
  }

}
