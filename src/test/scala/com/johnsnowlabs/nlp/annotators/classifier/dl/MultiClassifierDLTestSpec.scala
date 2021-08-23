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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest._

class MultiClassifierDLTestSpec extends FlatSpec {

  val spark = ResourceHelper.getActiveSparkSession

  "MultiClassifierDL" should "correctly train E2E Challenge" taggedAs SlowTest in {
    def splitAndTrim = udf { labels: String =>
      labels.split(", ").map(x=>x.trim)
    }

    val smallCorpus = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("mode", "DROPMALFORMED")
      .csv("src/test/resources/classifier/e2e.csv")
      .withColumn("labels", splitAndTrim(col("mr")))
      .drop("mr")

    println("count of training dataset: ", smallCorpus.count)
    smallCorpus.select("labels").show()
    smallCorpus.printSchema()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("ref")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val embeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("embeddings")

    val docClassifier = new MultiClassifierDLApproach()
      .setInputCols("embeddings")
      .setOutputCol("category")
      .setLabelColumn("labels")
      .setBatchSize(128)
      .setMaxEpochs(10)
      .setLr(1e-3f)
      .setThreshold(0.5f)
      .setValidationSplit(0.1f)
      .setRandomSeed(44)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          embeddings,
          docClassifier
        )
      )

    val pipelineModel = pipeline.fit(smallCorpus)

  }

}