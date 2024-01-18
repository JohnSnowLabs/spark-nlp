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

import com.johnsnowlabs.nlp.annotator.{SentenceDetectorDLModel, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class XlmRoBertaSentenceEmbeddingsTestSpec extends AnyFlatSpec {

  "XlmRoBertaSentenceEmbeddings" should "produce consistent embeddings" taggedAs SlowTest in {

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

    val embeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols(Array("sentence"))
      .setOutputCol("sentence_embeddings")
      .setMaxSentenceLength(32)

    val pipeline = new Pipeline().setStages(Array(document, sentence, embeddings))

    val model = pipeline.fit(testData)
    val results = model.transform(testData).select("id", "sentence_embeddings").collect()

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
      .asInstanceOf[XlmRoBertaSentenceEmbeddings]
      .write
      .overwrite()
      .save("./tmp_sent_xlm_roberta_base")

    val loadedEmbeddings = XlmRoBertaSentenceEmbeddings.load("./tmp_sent_xlm_roberta_base")
    val pipeline2 = new Pipeline().setStages(Array(document, sentence, loadedEmbeddings))

    val model2 = pipeline2.fit(testData)
    model2.transform(testData).select("id", "sentence_embeddings").show()
  }

  "XlmRoBertaSentenceEmbeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

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

    val embeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(32)

    val pipeline = new Pipeline().setStages(Array(document, sentence, embeddings))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select("sentence_embeddings.embeddings").show()

  }

  "XlmRoBertaSentenceEmbeddings" should "benchmark test" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save XlmRoBertaSentenceEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_xlm_roberta_sentence_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF
      .withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("sentence_embeddings")))
      .where(col("sentence_size") =!= col("embed_size"))
      .select(
        "sentence_size",
        "token_size",
        "embed_size",
        "token.result",
        "sentence_embeddings.result")
      .show(false)

    val totalSentences = pipelineDF.select(explode($"sentence.result")).count.toInt
    val totalEmbeddings =
      pipelineDF.select(explode($"sentence_embeddings.embeddings")).count.toInt

    println(s"total sentences: $totalSentences")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalSentences == totalEmbeddings)
  }

  "XlmRoBertaSentenceEmbeddings" should "download, save, and load a model" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_xlm_roberta_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_xlm_roberta_pipeline")

    pipelineModel.transform(ddd).show()
  }

  "XlmRoBertaSentenceEmbeddings" should "work with onnx" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = XlmRoBertaSentenceEmbeddings
      .loadSavedModel("onnx_models/xlm-roberta-base", ResourceHelper.spark)
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_xlm_roberta_sent_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_xlm_roberta_sent_pipeline")

    pipelineModel.transform(ddd).show()
  }
}
