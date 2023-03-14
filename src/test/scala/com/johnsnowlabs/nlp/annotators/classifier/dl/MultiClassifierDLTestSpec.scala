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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator.BertSentenceEmbeddings
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec.AnyFlatSpec

class MultiClassifierDLTestSpec extends AnyFlatSpec {

  val spark: SparkSession = ResourceHelper.getActiveSparkSession
  import spark.implicits._

  "MultiClassifierDL" should "correctly train E2E Challenge" taggedAs SlowTest in {
    def splitAndTrim = udf { labels: String =>
      labels.split(", ").map(x => x.trim)
    }

    val smallCorpus = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .option("mode", "DROPMALFORMED")
      .csv("src/test/resources/classifier/e2e.csv")
      .withColumn("labels", splitAndTrim(col("mr")))
      .drop("mr")

    println("count of training dataset: ", smallCorpus.count)
    smallCorpus.select("labels").show(1)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("ref")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val sentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("document")
      .setOutputCol("embeddings")

    val docClassifier = new MultiClassifierDLApproach()
      .setInputCols("embeddings")
      .setOutputCol("category")
      .setLabelColumn("labels")
      .setBatchSize(8)
      .setMaxEpochs(1)
      .setLr(1e-3f)
      .setThreshold(0.5f)
      .setValidationSplit(0.1f)
      .setRandomSeed(44)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceEmbeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
    pipelineModel.transform(smallCorpus).show(1)

  }

  "MultiClassifierDLApproach" should "not fail on empty validation sets" taggedAs SlowTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("document")
      .setOutputCol("embeddings")

    val docClassifier = new MultiClassifierDLApproach()
      .setInputCols("embeddings")
      .setOutputCol("category")
      .setLabelColumn("labels")
      .setBatchSize(8)
      .setMaxEpochs(1)
      .setLr(1e-3f)
      .setThreshold(0.5f)
      .setEnableOutputLogs(true)
      .setRandomSeed(44)
      .setValidationSplit(0.1f)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceEmbeddings, docClassifier))

    val data = Seq(
      ("This is good.", Array("good")),
      ("This is bad.", Array("bad")),
      ("This has no labels", Array.empty[String])).toDF("text", "labels")

    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).show(1)
  }

}
