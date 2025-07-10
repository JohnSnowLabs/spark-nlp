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
package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class DataFrameOptimizerTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "DataFrameOptimizer" should "optimize DataFrame operations" taggedAs FastTest in {
    val testDf = spark.createDataFrame(Seq((1, "test"), (2, "example")))
    testDf.show()
    val dataFrameOptimizer = new DataFrameOptimizer()
      .setExecutorCores(4)
      .setNumWorkers(1)
      .setDoCache(true)

    val pipeline = new Pipeline()
      .setStages(Array(dataFrameOptimizer))

    val optimizedDf = pipeline.fit(testDf).transform(testDf)

    optimizedDf.show()
  }

  it should "keep partitioning for downstream tasks" taggedAs FastTest in {
    val testDf = Seq("""This is a test sentence. It contains multiple sentences.""").toDF("text")
    val executorCores = 4
    val numWorkers = 2
    val dataFrameOptimizer = new DataFrameOptimizer()
      .setExecutorCores(executorCores)
      .setNumWorkers(numWorkers)
      .setDoCache(true)

    val pipeline = new Pipeline()
      .setStages(Array(dataFrameOptimizer, documentAssembler, sentenceDetector))

    val optimizedResultDf = pipeline.fit(testDf).transform(testDf)
    assert(
      optimizedResultDf.rdd.getNumPartitions == numWorkers * executorCores,
      "DataFrame should be partitioned for downstream tasks")
  }

  it should "persist DataFrame to disk" taggedAs SlowTest in {
    val testDf = spark.createDataFrame(Seq((1, "test"), (2, "example")))
    val persistPath = "./tmp_DataFrameOptimizer"
    val dataFrameOptimizer = new DataFrameOptimizer()
      .setExecutorCores(4)
      .setNumWorkers(1)
      .setDoCache(true)
      .setPersistPath(persistPath)
      .setPersistFormat("parquet")

    val pipeline = new Pipeline()
      .setStages(Array(dataFrameOptimizer))

    pipeline.fit(testDf).transform(testDf)

    assert(spark.read.parquet(persistPath).count() == 2, "DataFrame should be persisted to disk")
  }

}
