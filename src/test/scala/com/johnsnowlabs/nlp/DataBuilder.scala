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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


object DataBuilder extends FlatSpec with BeforeAndAfterAll { this: Suite =>

  import SparkAccessor.spark.implicits._

  def basicDataBuild(content: String*)(implicit cleanupMode: String = "disabled"): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data, cleanupMode)
  }

  def multipleDataBuild(content: Seq[String]): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def buildNerDataset(datasetContent: String): Dataset[Row] = {
    val lines = datasetContent.split("\n")
    val data = CoNLL(conllLabelIndex = 1)
      .readDatasetFromLines(lines, SparkAccessor.spark).toDF
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def loadParquetDataset(path: String) =
    SparkAccessor.spark.read.parquet(path)
}
