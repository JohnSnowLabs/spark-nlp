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

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class XlmRoBertaForTokenClassificationTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "XlmRoBertaForTokenClassification" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

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

    val tokenClassifier = XlmRoBertaForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("token.result").show(false)
    pipelineDF.select("label.result").show(false)
    pipelineDF
      .withColumn("token_size", size(col("token")))
      .withColumn("label_size", size(col("label")))
      .where(col("token_size") =!= col("label_size"))
      .select("token_size", "label_size", "token.result", "label.result")
      .show(false)

    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"label.result")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

  }

  "XlmRoBertaForTokenClassification" should "be saved and loaded correctly" taggedAs SlowTest in {

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

    val tokenClassifier = XlmRoBertaForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label.result").show(false)

    Benchmark.time("Time to save XlmRoBertaForTokenClassification pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_xlmrobertafortoken_pipeline")
    }

    Benchmark.time("Time to save XlmRoBertaForTokenClassification model") {
      pipelineModel.stages.last
        .asInstanceOf[XlmRoBertaForTokenClassification]
        .write
        .overwrite()
        .save("./tmp_xlmrobertafortoken_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_xlmrobertafortoken_pipeline")
    loadedPipelineModel.transform(ddd).select("label.result").show(false)

    val loadedSequenceModel =
      XlmRoBertaForTokenClassification.load("./tmp_xlmrobertafortoken_model")
    println(loadedSequenceModel.getClasses.mkString("Array(", ", ", ")"))

  }

  "XlmRoBertaForTokenClassification" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
    val tokenClassifier = XlmRoBertaForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("ner")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(tokenClassifier))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save the results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_xlm_roberta_token_classifier")
    }

    println("missing tokens/tags: ")
    pipelineDF
      .withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("ner_size", size(col("ner")))
      .where(col("token_size") =!= col("ner_size"))
      .select("sentence_size", "token_size", "ner_size", "token.result", "ner.result")
      .show(false)

    println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalTags = pipelineDF.select(explode($"ner.result")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalTags")

    assert(totalTokens == totalTags)
  }
}
