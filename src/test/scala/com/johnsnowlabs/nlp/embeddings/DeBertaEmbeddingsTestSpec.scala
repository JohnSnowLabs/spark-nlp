@@ -1,162 +1,366 @@
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
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class DeBertaEmbeddingsTestSpec extends AnyFlatSpec {

  "DeBertaEmbeddings" should "correctly load pretrained model" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, embeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.select("token.result").show(1, truncate = false)
    pipelineDF.select("embeddings.result").show(1, truncate = false)
    pipelineDF.select("embeddings.metadata").show(1, truncate = false)
    pipelineDF.select("embeddings.embeddings").show(1, truncate = 300)
    pipelineDF.select(size(pipelineDF("embeddings.embeddings")).as("embeddings_size")).show(1)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.select("embeddings").write.mode("overwrite").parquet("./tmp_embeddings")
    }
  }

  "DeBertaEmbeddings" should "benchmark test" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save DeBertaEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

    Benchmark.time("Time to finish checking counts in results") {
      println("missing tokens/embeddings: ")
      pipelineDF
        .withColumn("sentence_size", size(col("sentence")))
        .withColumn("token_size", size(col("token")))
        .withColumn("embed_size", size(col("embeddings")))
        .where(col("token_size") =!= col("embed_size"))
        .select("sentence_size", "token_size", "embed_size")
        .show(false)
    }

    Benchmark.time("Time to finish explod/count in results") {
      println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
      val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
      val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

      println(s"total tokens: $totalTokens")
      println(s"total embeddings: $totalEmbeddings")

      assert(totalTokens == totalEmbeddings)

    }
  }

  "DeBertaEmbeddings" should "be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.").toDF(
      "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(128)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("token").show(false)
    pipelineDF.select("embeddings.result").show(false)
    pipelineDF
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)

  }

  "DeBertaEmbeddings" should "onnx be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.").toDF(
      "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    //    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base"
    //    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base_opt"
    //    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base_opt-quantized"

    //    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum"
    //    val modelPath =
    //      "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum/optimize-gpu-false"
    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum/quantize-arm64"

    val embeddings = DeBertaEmbeddings
//      .pretrained("deberta_v3_base")
      .loadSavedModel(modelPath, ResourceHelper.spark)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("token").show(false)
    pipelineDF.select("embeddings.result").show(false)
    pipelineDF
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)

  }

  "DeBertaEmbeddings" should "onnx" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth. .  . .",
      "\\u2009.carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val modelPath =
      "/Users/maziyar/Downloads/bert_onnx_export/bert_base_cased_convert_graph_to_onnx_quantize/"

    val embeddings = BertEmbeddings
      //      .pretrained("small_bert_L2_128")
      .loadSavedModel(modelPath, ResourceHelper.spark)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(12)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)

    pipelineModel.stages.last
      .asInstanceOf[BertEmbeddings]
      .write
      .overwrite()
      .save("./tmp_bert_onnx_model")

    val pipelineDF = pipelineModel.transform(ddd)

    println("tokens: ")
    pipelineDF.select("token.result").show(false)
    println("embeddings: ")
    pipelineDF.select("embeddings.result").show(false)

    val loadedBertModel = BertEmbeddings.load("./tmp_bert_onnx_model")
    val pipeline2 = new Pipeline().setStages(Array(document, tokenizer, loadedBertModel))

    val pipelineDF2 = pipeline2.fit(ddd).transform(ddd)

    pipelineDF.select("embeddings.embeddings").show()
    pipelineDF2.select("embeddings.embeddings").show()

    pipelineDF2
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    val totalTokens = pipelineDF2.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF2.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)

  }

  "DeBertaEmbeddings" should "tf vs onnx benchmark test" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")

//    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base"
//    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base_opt"
//    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-v3-base_opt-quantized"

//    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum"
//    val modelPath =
//      "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum/optimize-gpu-false"
//    val modelPath = "/Users/maziyar/Downloads/onnx_export/deberta/deberta-optimum/quantize-arm64"

    val embeddings = DeBertaEmbeddings
      .pretrained("deberta_v3_base")
//      .loadSavedModel(modelPath, ResourceHelper.spark)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    //    warmup
    val pipeline = new Pipeline()
      .setStages(Array(embeddings))
    Benchmark.measure(iterations = 10, forcePrint = true, description = "Time to warmup") {
      pipeline.fit(training_data).transform(training_data.limit(1)).count()
    }

    val pipelineModel = pipeline.fit(training_data)
    val pipelineLightPipeline = new LightPipeline(pipelineModel)
    val str = "carbon emissions have come down without impinging on our growth."

    Benchmark.measure(
      iterations = 10,
      forcePrint = true,
      description =
        s"Time to process 1 string in LightPipeline with sequence length: ${str.length}") {
      pipelineLightPipeline.annotate(str)
    }

    //    Array(2, 4, 8, 16, 32).foreach(b => {
    Array(1, 4, 8, 16).foreach(b => {
//    Array(4).foreach(b => {
      embeddings.setBatchSize(b)

      val pipeline = new Pipeline()
        .setStages(Array(embeddings))

      val pipelineModel = pipeline.fit(training_data)
      val pipelineDF = pipelineModel.transform(training_data)

      Benchmark.measure(
        iterations = 1,
        forcePrint = true,
        description =
          s"Time to save pipeline with batch size: ${pipelineModel.stages(0).asInstanceOf[DeBertaEmbeddings].getBatchSize}") {
        pipelineDF.write.mode("overwrite").parquet("./tmp_use_sentence_embeddings")
      }
    })

    /* tf

     */
  }
}