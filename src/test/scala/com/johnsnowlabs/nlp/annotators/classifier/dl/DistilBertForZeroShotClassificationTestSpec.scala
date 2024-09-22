/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class DistilBertForZeroShotClassificationTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  val candidateLabels =
    Array("urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology")

  "DistilBertForZeroShotClassification" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val ddd = Seq(
      "I have a problem with my iphone that needs to be resolved asap!!",
      "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.",
      "I have a phone and I love it!",
      "I really want to visit Germany and I am planning to go there next year.",
      "Let's watch some movies tonight! I am in the mood for a horror movie.",
      "Have you watched the match yesterday? It was a great game!",
      "We need to harry up and get to the airport. We are going to miss our flight!")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tokenClassifier = DistilBertForZeroShotClassification
      .loadSavedModel("1",ResourceHelper.spark)
      .setInputCols(Array("token", "document"))
      .setOutputCol("multi_class")
      .setCaseSensitive(true)
      .setCoalesceSentences(true)
      .setCandidateLabels(candidateLabels)
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("multi_class").show(20, false)
    pipelineDF.select("document.result", "multi_class.result").show(20, false)
    pipelineDF
      .withColumn("doc_size", size(col("document")))
      .withColumn("label_size", size(col("multi_class")))
      .where(col("doc_size") =!= col("label_size"))
      .select("doc_size", "label_size", "document.result", "multi_class.result")
      .show(20, false)

    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"multi_class.result")).count.toInt

    println(s"total tokens: $totalDocs")
    println(s"total embeddings: $totalLabels")

    assert(totalDocs == totalLabels)
  }

  "DistilBertForZeroShotClassification" should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tokenClassifier = DistilBertForZeroShotClassification
      .loadSavedModel("1",ResourceHelper.spark)
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)
      .setCoalesceSentences(true)
      .setCandidateLabels(candidateLabels)
      .setBatchSize(2)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label.result").show(false)

    Benchmark.time("Time to save DistilBertForZeroShotClassification pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_distilbertfornli_pipeline")
    }

    Benchmark.time("Time to save DistilBertForZeroShotClassification model") {
      pipelineModel.stages.last
        .asInstanceOf[DistilBertForZeroShotClassification]
        .write
        .overwrite()
        .save("./tmp_distilbertfornli_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_distilbertfornli_pipeline")
    loadedPipelineModel.transform(ddd).select("label.result").show(false)

    val loadedSequenceModel =
      DistilBertForZeroShotClassification.load("./tmp_distilbertfornli_model")
    println(loadedSequenceModel.getClasses.mkString("Array(", ", ", ")"))

  }

  "DistilBertForZeroShotClassification" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll
        .readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
        .repartition(12)
        .limit(30)

    val tokenClassifier = DistilBertForZeroShotClassification
      .loadSavedModel("1",ResourceHelper.spark)
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setCoalesceSentences(true)
      .setCandidateLabels(candidateLabels)
      .setBatchSize(2)

    val pipeline = new Pipeline()
      .setStages(Array(tokenClassifier))

    val pipelineDF = pipeline.fit(training_data).transform(training_data).cache()
    Benchmark.time("Time to save pipeline results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_nli_classifier")
    }

    pipelineDF.select("class").show(2, false)
    pipelineDF.select("sentence.result", "class.result").show(2, false)

    // only works if it's softmax - one lablel per row
    pipelineDF
      .withColumn("doc_size", size(col("sentence")))
      .withColumn("label_size", size(col("class")))
      .where(col("doc_size") =!= col("label_size"))
      .select("doc_size", "label_size", "sentence.result", "class.result")
      .show(20, false)

    val totalDocs = pipelineDF.select(explode($"sentence.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"class.result")).count.toInt

    println(s"total docs: $totalDocs")
    println(s"total classes: $totalLabels")

    assert(totalDocs == totalLabels)
  }
}
