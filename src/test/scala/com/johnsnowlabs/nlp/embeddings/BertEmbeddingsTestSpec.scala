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

import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class BertEmbeddingsTestSpec extends AnyFlatSpec {

  "Bert Embeddings" should "correctly embed tokens and sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val data1 = Seq(
      "In the Seven Kingdoms of Westeros, a soldier of the ancient Night's Watch order survives an attack by supernatural creatures known as the White Walkers, thought until now to be mythical.")
      .toDF("text")

    val data2 = Seq(
      "In King's Landing, the capital, Jon Arryn, the King's Hand, dies under mysterious circumstances.")
      .toDF("text")

    val data3 = Seq(
      "Tyrion makes saddle modifications for Bran that will allow the paraplegic boy to ride.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128", "en")
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).transform(ddd)
    pipeline.fit(data1).transform(data1)
    pipeline.fit(data2).transform(data2)
    pipeline.fit(data3).transform(data3)

  }

  "Bert Embeddings" should "onnx" taggedAs SlowTest in {

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
      "/home/maziyar/data/onnx_export/bert_base_cased_onnxruntime_opt/"

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

  "bert" should "tf vs onnx benchmark test" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val conll = CoNLL()
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    //    val modelPath = "/home/maziyar/data/onnx_export/transformer_cli_onnx/"
    //    val modelPath = "/home/maziyar/data/onnx_export/torch_onnx_export/"
    //    val modelPath =
    //      "/home/maziyar/data/onnx_export/optimum_ORTModelForFeatureExtraction_onnx/"
    //    val modelPath =
    //      "/home/maziyar/data/onnx_export/optimum_ORTModelForFeatureExtraction_ORTQuantizer_onnx"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_optimize-gpu-false/"
//    val modelPath = "/home/maziyar/data/onnx_export/optimum_optimize-gpu-true/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx2/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx512/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx512_vnni/"
//    val modelPath = "/home/maziyar/data/onnx_export/bert_base_cased_onnxruntime_opt"
//    val modelPath =
//      "/home/maziyar/data/onnx_export/bert_base_cased_convert_graph_to_onnx_quantize/"

//    val modelPath = "/home/maziyar/data/onnx_export/torch32/"
    val modelPath = "/home/maziyar/data/onnx_export/ORTModelForFeatureExtraction/"

    val embeddings = BertEmbeddings
//      .pretrained("bert_base_cased")
      .loadSavedModel(modelPath, ResourceHelper.spark)
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

    //    Array(2, 4, 8, 16, 32).foreach(b => {
//    Array(1, 2, 4, 8).foreach(b => {
    Array(4).foreach(b => {
      embeddings.setBatchSize(b)

      val pipeline = new Pipeline()
        .setStages(Array(embeddings))

      val pipelineModel = pipeline.fit(training_data)
      val pipelineLightPipeline = new LightPipeline(pipelineModel)
      val pipelineDF = pipelineModel.transform(training_data)

      val str = "carbon emissions have come down without impinging on our growth."
      val strDF =
        Seq("carbon emissions have come down without impinging on our growth.").toDF("text")

      println(s"batch size: ${pipelineModel.stages(0).asInstanceOf[BertEmbeddings].getBatchSize}")
      println(s"sequence length: ${str.length}")

      Benchmark.measure(
        iterations = 5,
        forcePrint = true,
        description = "Time to process 1 string in LightPipeline") {
        pipelineLightPipeline.annotate(str)
      }

      Benchmark.measure(
        iterations = 2,
        forcePrint = true,
        description = "Time to save pipeline") {
        pipelineDF.write.mode("overwrite").parquet("./tmp_use_sentence_embeddings")
      }
    })
    /*
    ORTModelForFeatureExtraction
    Time to warmup (Avg for 10 iterations): 1.1380769792 sec
    batch size: 4
    sequence length: 64
    Time to process 1 string in LightPipeline (Avg for 5 iterations): 7.499462E-4 sec
    Time to save pipeline (Avg for 2 iterations): 83.653844983 sec

     */

  }

  "bert" should "tf vs onnx LP benchmark test" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._
    val testText: String =
      """William Henry Gates III (born October 28, 1955) is an American business magnate, software developer,
        |investor,and philanthropist. He is best known as the co-founder of Microsoft Corporation.
        |During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO),
        |president and chief software architect, while also being the largest individual shareholder until May 2014.
        |He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s.""".stripMargin

    val strDF = Seq(testText).toDF("text")
    val test100 = Array.fill(100)(testText)
    val test1000 = Array.fill(1000)(testText)
    val test10000 = Array.fill(10000)(testText)

    val docInputCol = "text"
    val docOutputCol = "document"
    val sentOutputCol = "sentence"
    val tokenOutputCol = "token"
    val embeddingsOutputCol = "embeddings"
    val nerOutputCol = "ner"
    val nerConverterOutputCol = "entities"

    val document: DocumentAssembler = new DocumentAssembler()
      .setInputCol(docInputCol)
      .setOutputCol(docOutputCol)

    val sentence: SentenceDetector = new SentenceDetector()
      .setInputCols(docOutputCol)
      .setOutputCol(sentOutputCol)

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(sentOutputCol)
      .setOutputCol(tokenOutputCol)

    val entity: NerConverter = new NerConverter()
      .setInputCols(sentOutputCol, tokenOutputCol, nerOutputCol)
      .setOutputCol(nerConverterOutputCol)

    //    val modelPath = "/home/maziyar/data/onnx_export/transformer_cli_onnx/"
    //    val modelPath = "/home/maziyar/data/onnx_export/torch_onnx_export/"
    //    val modelPath =
    //      "/home/maziyar/data/onnx_export/optimum_ORTModelForFeatureExtraction_onnx/"
    //    val modelPath =
    //      "/home/maziyar/data/onnx_export/optimum_ORTModelForFeatureExtraction_ORTQuantizer_onnx"
//    val modelPath = "/home/maziyar/data/onnx_export/optimum_optimize-gpu-false/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_optimize-gpu-true/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx2/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx512/"
    //    val modelPath = "/home/maziyar/data/onnx_export/optimum_quantize-avx512_vnni/"
    //    val modelPath = "/home/maziyar/data/onnx_export/bert_base_cased_onnxruntime_opt"
    val modelPath =
      "/home/maziyar/data/onnx_export/bert_base_cased_convert_graph_to_onnx_quantize/"

    //    val modelPath = "/home/maziyar/data/onnx_export/bert_uncased_l2_h_128_a2"
//    val modelPath = "/home/maziyar/data/onnx_export/bert_uncased_l2_h_128_a2_opt"
//    val modelPath = "/home/maziyar/data/onnx_export/bert_uncased_l2_h_128_a2_opt-quantized"

    val embeddings = BertEmbeddings
//      .pretrained("bert_base_cased")
//      .load("/Users/maziyar/cache_pretrained/small_bert_L2_128_en_2.6.0_2.4_1598344320681")
      .loadSavedModel(modelPath, ResourceHelper.spark)
//      .setStorageRef("small_bert_L2_128")
//      .setStorageRef("bert_base_cased")
      .setInputCols("sentence", "token")
      .setOutputCol(embeddingsOutputCol)
      .setMaxSentenceLength(512)
      .setCaseSensitive(true)
      .setBatchSize(1)

    val nerOnto = NerDLModel
      .pretrained("onto_bert_base_cased", "en")
//      .load("/Users/maziyar/cache_pretrained/onto_small_bert_L2_128_en_2.7.0_2.4_1607198998042")
      .setInputCols(Array("sentence", "token", embeddingsOutputCol))
      .setOutputCol("ner")
      .setIncludeConfidence(true)
      .setBatchSize(10000)

//    Array(1, 2, 4, 8).foreach(b => {
//        Array(1, 2, 4, 8).foreach(b => {
    Array(1).foreach(b => {
      embeddings.setBatchSize(b)

      val pipeline =
        new Pipeline().setStages(
          Array(document, sentence, tokenizer, embeddings, nerOnto, entity))

      val pipelineModel = pipeline.fit(strDF)
      val pipelineLightPipeline = new LightPipeline(pipelineModel)

      println(s"batch size: ${pipelineModel.stages(3).asInstanceOf[BertEmbeddings].getBatchSize}")
      println(s"sequence length: ${testText.length}")
      val res = pipelineLightPipeline.annotate(testText)
      println(res("entities"))

      Benchmark.measure(iterations = 30, forcePrint = true, description = "Time to warmup") {
        pipelineLightPipeline.fullAnnotate(Array(testText))
      }

      Benchmark.measure(
        iterations = 5,
        forcePrint = true,
        description = "Time to process 1 string in LightPipeline") {
        pipelineLightPipeline.fullAnnotate(Array(testText))
      }

      Benchmark.measure(
        iterations = 5,
        forcePrint = true,
        description = "Time to process 100 string in LightPipeline") {
        pipelineLightPipeline.fullAnnotate(test100)
      }

      Benchmark.measure(
        iterations = 5,
        forcePrint = true,
        description = "Time to process 1000 string in LightPipeline") {
        pipelineLightPipeline.fullAnnotate(test1000)
      }

//      Benchmark.measure(
//        iterations = 5,
//        forcePrint = true,
//        description = "Time to process 10000 string in LightPipeline") {
//        pipelineLightPipeline.fullAnnotate(test10000)
//      }

      /*

      TF
      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, the microcomputer revolution, 1970s, 1980s)
      Time to warmup (Avg for 20 iterations): 0.03809238745 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.032690989600000005 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 0.6901927967999999 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 7.9315060248 sec

      ONNX
      bert_uncased_l2_h_128_a2

      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, the microcomputer revolution, 1970s, 1980s)
      Time to warmup (Avg for 20 iterations): 0.03567430395 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.028602725199999998 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 0.674258651 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 7.4446835048 sec

      bert_uncased_l2_h_128_a2_opt

      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, the microcomputer revolution, 1970s, 1980s)
      Time to warmup (Avg for 20 iterations): 0.03335386455 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.024432008600000003 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 0.6897019274 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 7.3714859024 sec


      bert_uncased_l2_h_128_a2_opt-quantized

      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, the microcomputer revolution, 1970s, 1980s)
      Time to warmup (Avg for 20 iterations): 0.0335167508 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.025876901 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 0.681434737 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 7.5693501748 sec


      optimum_optimize-gpu-false

      batch size: 1
      sequence length: 510
      ArraySeq(October 28, 1955, American, 2014, one, the 1970s and 1980s)
      Time to warmup (Avg for 20 iterations): 0.1453052615 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.1310928428 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 10.17499


      bert-base-cased

      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, one, the 1970s and 1980s)
      Time to warmup (Avg for 20 iterations): 0.23966487185 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.2463635468 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 16.5812129422 sec

      bert-base-cased AMD:
      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, one, the 1970s and 1980s)
      Time to warmup (Avg for 20 iterations): 0.26150889755 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.252887444 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 7.106950422 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 56.6557830706 sec


      optimum_optimize-gpu-false
      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, one, the 1970s and 1980s)
      Time to warmup (Avg for 20 iterations): 0.08651849975 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.08021795579999999 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 4.2544900956 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 40.2780014848 sec


      bert_base_cased_convert_graph_to_onnx_quantize
      batch size: 1
      sequence length: 510
      ArraySeq(William Henry Gates III, October 28, 1955, American, Microsoft Corporation, Microsoft, Gates, May 2014, one, the 1970s and 1980s)
      Time to warmup (Avg for 30 iterations): 0.0890891555 sec
      Time to process 1 string in LightPipeline (Avg for 5 iterations): 0.0862840258 sec
      Time to process 100 string in LightPipeline (Avg for 5 iterations): 2.336687043 sec
      Time to process 1000 string in LightPipeline (Avg for 5 iterations): 18.6741625948 sec

       */
    })

  }
}
