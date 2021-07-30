/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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
import org.scalatest._

class BertForTokenClassificationTestSpec extends FlatSpec {

  import ResourceHelper.spark.implicits._

  "BertForTokenClassification" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21"
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tokenClassifier = BertForTokenClassification.pretrained()
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

  "BertForTokenClassification" should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21"
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tokenClassifier = BertForTokenClassification.pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label.result").show(false)

    Benchmark.time("Time to save BertForTokenClassification pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_bertfortoken_pipeline")
    }

    Benchmark.time("Time to save BertForTokenClassification model") {
      pipelineModel.stages.last.asInstanceOf[BertForTokenClassification].write.overwrite().save("./tmp_bertfortoken_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_bertfortoken_pipeline")
    loadedPipelineModel.transform(ddd).select("label.result").show(false)

    val loadedDistilBertModel = BertForTokenClassification.load("./tmp_bertfortoken_model")
    loadedDistilBertModel.getLabels

  }

  "BertForTokenClassification" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val tokenClassifier = BertForTokenClassification.pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        tokenClassifier
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save BertForTokenClassification results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF.withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("sentence_size", "token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)
  }
}
