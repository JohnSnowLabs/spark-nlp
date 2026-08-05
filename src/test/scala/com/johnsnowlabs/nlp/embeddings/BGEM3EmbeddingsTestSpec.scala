/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, size}
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Try

class BGEM3EmbeddingsTestSpec extends AnyFlatSpec {

  private def parseableAsFloat(s: String): Boolean = Try(s.toFloat).isSuccess

  /** Structural metadata keys inherited from upstream annotators / added by the embeddings
    * wrapper. These are excluded when inspecting the sparse lexical weights.
    */
  private val structuralKeys = Set("sentence", "id", "token", "pieceId", "isWordStart", "isOOV")

  private def sparseWeightsOf(
      metadata: scala.collection.Map[String, String]): scala.collection.Map[String, String] =
    metadata.filter { case (k, v) => !structuralKeys.contains(k) && parseableAsFloat(v) }

  "BGE-M3 Embeddings" should "correctly embed multilingual sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "How much protein should a female eat?",
      "¿Cuánta proteína debería comer una mujer?",
      "Combien de protéines une femme devrait-elle manger ?",
      "女性はどのくらいのタンパク質を摂取すべきですか？",
      "امرأة كم من البروتين يجب أن تأكل؟")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEM3Embeddings
      .pretrained("bge_m3", "xx")
      .setInputCols(Array("document"))
      .setOutputCol("bge_m3")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("bge_m3.embeddings").show(truncate = false)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("bge_m3.embeddings").getItem(0))
    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    // Dense embeddings should all be the same (1024) dimension and non-empty
    assert(sizesArray.forall(_ == 1024))
  }

  it should "produce sparse lexical weights in the metadata when enabled" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("BGE-M3 supports both dense and sparse retrieval.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEM3Embeddings
      .pretrained("bge_m3", "xx")
      .setInputCols(Array("document"))
      .setOutputCol("bge_m3")
      .setReturnSparseEmbeddings(true)

    val pipeline = new Pipeline().setStages(Array(document, embeddings))
    val pipelineDF = pipeline.fit(ddd).transform(ddd)

    val annotations: Seq[Annotation] = Annotation.collect(pipelineDF, "bge_m3").head.toSeq
    val metadata = annotations.head.metadata

    // At least a few {token: weight} pairs should be present and parseable as floats.
    val sparseWeights = sparseWeightsOf(metadata)
    assert(sparseWeights.nonEmpty, "Expected sparse lexical weights in the annotation metadata")
    assert(sparseWeights.values.forall(_.toFloat > 0f), "Sparse weights should be positive")

    // Dense embedding is still present
    assert(annotations.head.embeddings.length == 1024)
  }

  it should "not compute sparse weights by default (dense only)" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("A dense-only pipeline should not pay for the sparse head.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEM3Embeddings
      .pretrained("bge_m3", "xx")
      .setInputCols(Array("document"))
      .setOutputCol("bge_m3")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))
    val pipelineDF = pipeline.fit(ddd).transform(ddd)

    val annotations: Seq[Annotation] = Annotation.collect(pipelineDF, "bge_m3").head.toSeq
    val sparseWeights = sparseWeightsOf(annotations.head.metadata)

    assert(sparseWeights.isEmpty, "No sparse weights should be present when disabled")
    assert(annotations.head.embeddings.length == 1024)
  }

  it should "handle long documents close to the 8192 token ceiling" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val longText = (1 to 2000)
      .map(i => s"Sentence number $i talks about multilingual retrieval and embeddings.")
      .mkString(" ")

    val ddd = Seq(longText).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEM3Embeddings
      .pretrained("bge_m3", "xx")
      .setInputCols(Array("document"))
      .setOutputCol("bge_m3")
      .setMaxSentenceLength(8192)

    val pipeline = new Pipeline().setStages(Array(document, embeddings))
    val pipelineDF = pipeline.fit(ddd).transform(ddd)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("bge_m3.embeddings").getItem(0))
    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ == 1024))
  }

  it should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Dense and sparse embeddings from a single annotator.",
      "Modelo multilingüe de incrustaciones.").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEM3Embeddings
      .pretrained("bge_m3", "xx")
      .setInputCols(Array("document"))
      .setOutputCol("embeddings")
      .setReturnSparseEmbeddings(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    pipelineModel.transform(ddd).select("embeddings.result").show(false)

    Benchmark.time("Time to save BGEM3Embeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_bge_m3_pipeline")
    }

    Benchmark.time("Time to save BGEM3Embeddings model") {
      pipelineModel.stages.last
        .asInstanceOf[BGEM3Embeddings]
        .write
        .overwrite()
        .save("./tmp_bge_m3_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_bge_m3_pipeline")
    loadedPipelineModel.transform(ddd).select("embeddings.result").show(false)

    val loadedModel = BGEM3Embeddings.load("./tmp_bge_m3_model")
    assert(loadedModel.getReturnSparseEmbeddings)
  }

}
