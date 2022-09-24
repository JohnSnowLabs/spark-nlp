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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class CamemBertForTokenClassificationTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  val ddd: DataFrame = Seq(
    "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
    "Rare Hendrix song draft sells for almost $17,000.",
    "EU rejects German call to boycott British lamb .",
    "TORONTO 1996-08-21",
    "Barack Obama /bəˈɹɑːk oʊˈbɑːmə/, né le 4 août 1961 à Honolulu (Hawaï), est un homme d'État américain. Il est le 44e président des États-Unis",
    "Paris est la capitale de la France.",
    "george washington est allé à washington",
    "\\u2009.Les modèles de langue contextuels Camembert pour le Français : impact de la taille et de l'hétérogénéité des données d'entrainement .\\u2009.\\u2009.")
    .toDF("text")

  val document: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  "CamemBertForTokenClassification" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val tokenClassifier: CamemBertForTokenClassification = CamemBertForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("ner")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("token.result").show(false)
    pipelineDF.select("ner.result").show(false)
    pipelineDF
      .withColumn("token_size", size(col("token")))
      .withColumn("ner_size", size(col("ner")))
      .where(col("token_size") =!= col("ner_size"))
      .select("token_size", "ner_size", "token.result", "ner.result")
      .show(false)

    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"ner.result")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

  }

  "CamemBertForTokenClassification" should "be saved and loaded correctly" taggedAs SlowTest in {

    val tokenClassifier: CamemBertForTokenClassification = CamemBertForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("ner")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, tokenClassifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("ner.result").show(false)

    Benchmark.time("Time to save CamemBertForTokenClassification pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_fortoken_pipeline")
    }

    Benchmark.time("Time to save BertForTokenClassification model") {
      pipelineModel.stages.last
        .asInstanceOf[CamemBertForTokenClassification]
        .write
        .overwrite()
        .save("./tmp_fortoken_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_fortoken_pipeline")
    loadedPipelineModel.transform(ddd).select("ner.result").show(false)

    val loadedSequenceModel =
      CamemBertForTokenClassification.load("./tmp_fortoken_model")
    println(loadedSequenceModel.getClasses.mkString("Array(", ", ", ")"))

  }

  "CamemBertForTokenClassification" should "benchmark test" taggedAs SlowTest in {

    val tokenClassifier: CamemBertForTokenClassification = CamemBertForTokenClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("ner")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")

    val pipeline = new Pipeline()
      .setStages(Array(tokenClassifier))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save the results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_token_classifier")
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
