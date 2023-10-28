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

package com.johnsnowlabs.nlp.embeddings

 import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class BertSentenceEmbeddingsTestSpec extends AnyFlatSpec {

  "BertSentenceEmbeddings" should "produce consistent embeddings" taggedAs FastTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq((1, "John loves apples."), (2, "Mary loves oranges. John loves Mary.")))
      .toDF("id", "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols(Array("sentence"))
      .setOutputCol("bert")
      .setMaxSentenceLength(32)

    val pipeline = new Pipeline().setStages(Array(document, sentence, embeddings))

    val model = pipeline.fit(testData)
    val results = model.transform(testData).select("id", "bert").collect()

    results.foreach(row => {
      val rowI = row.get(0)
      row
        .get(1)
        .asInstanceOf[mutable.WrappedArray[GenericRowWithSchema]]
        .zipWithIndex
        .foreach(t => {
          print(
            "%1$-3s\t%2$-3s\t%3$-30s\t".format(rowI.toString, t._2.toString, t._1.getString(3)))
          println(
            t._1
              .get(5)
              .asInstanceOf[mutable.WrappedArray[Float]]
              .slice(0, 5)
              .map("%1$-7.3f".format(_))
              .mkString(" "))
        })
    })

    model
      .stages(2)
      .asInstanceOf[BertSentenceEmbeddings]
      .write
      .overwrite()
      .save("./tmp_bert_sentence_embed")
  }

  "BertSentenceEmbeddings" should "correctly work with empty tokens" taggedAs FastTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (1, "This is my first sentence. This is my second."),
          (2, "This is my third sentence. This is my forth.")))
      .toDF("id", "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("sentence")
      .setOutputCol("bert")
      .setCaseSensitive(false)
      .setMaxSentenceLength(32)

    val pipeline = new Pipeline().setStages(Array(document, sentence, embeddings))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select("bert.embeddings").show()

  }

  "BertSentenceEmbeddings" should "benchmark test" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("sentence")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save BertSentenceEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_sentence_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF
      .withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("sentence_size") =!= col("embed_size"))
      .select("sentence_size", "token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    val totalSentences = pipelineDF.select(explode($"sentence.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total sentences: $totalSentences")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalSentences == totalEmbeddings)
  }

  "BertSentenceEmbeddings" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tfModelPath = "src/test/resources/tf-hub-bert/model"

    val embeddings = BertSentenceEmbeddings
      .loadSavedModel(tfModelPath, ResourceHelper.spark)
      .setInputCols("document")
      .setOutputCol("bert")
      .setStorageRef("tf_hub_bert_test")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_bert_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_bert_pipeline")

    pipelineModel.transform(ddd)
  }

  "BertSentenceEmbeddings" should "correctly propagate metadata" taggedAs FastTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(Seq((
        1,
        "\"My first sentence with the first rule. This is my second sentence with ceremonies rule")))
      .toDF("id", "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexMatcher = new RegexMatcher()
      .setExternalRules(
        ExternalResource(
          "src/test/resources/regex-matcher/rules.txt",
          ReadAs.TEXT,
          Map("delimiter" -> ",")))
      .setInputCols(Array("sentence"))
      .setOutputCol("regex")
      .setStrategy("MATCH_ALL")

    val chunk2Doc = new Chunk2Doc().setInputCols("regex").setOutputCol("doc_chunk")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("doc_chunk")
      .setOutputCol("bert")
      .setCaseSensitive(false)
      .setMaxSentenceLength(32)

    val pipeline =
      new Pipeline().setStages(Array(document, sentence, regexMatcher, chunk2Doc, embeddings))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select("bert.metadata").show(truncate = false)

  }

  it should "work with onnx" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val path = "onnx_models/bert-base-cased"
    val embeddings = BertSentenceEmbeddings
      .loadSavedModel(path, ResourceHelper.spark)
      .setInputCols("document")
      .setOutputCol("bert")
      .setMaxSentenceLength(512)

    embeddings.write
      .overwrite()
      .save("bert_sent_onnx")

    val loadedEmbeddings = BertSentenceEmbeddings.load("bert_sent_onnx")

    val pipeline = new Pipeline().setStages(Array(document, loadedEmbeddings))

    pipeline.fit(ddd).transform(ddd).select("bert").show(truncate = false)
  }
}
