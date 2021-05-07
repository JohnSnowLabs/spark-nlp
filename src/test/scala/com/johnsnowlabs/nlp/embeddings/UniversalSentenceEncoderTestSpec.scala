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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.EmbeddingsFinisher
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel, Normalizer, SQLTransformer}
import org.apache.spark.sql.functions._

import org.scalatest._

class UniversalSentenceEncoderTestSpec extends FlatSpec {

  "UniversalSentenceEncoder" should "correctly calculate sentence embeddings for a sentence" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings_use.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        useEmbeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    println(pipelineDF.count())
    Benchmark.time("Time to save USE results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_use_embeddings")
    }

  }

  "UniversalSentenceEncoder" should "integrate into Spark ML" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings_use.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("sentence_embeddings_vectors")
      .setCleanAnnotations(false)
      .setOutputAsVector(true)

    val explodeVectors = new SQLTransformer().setStatement(
      "SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")

    val vectorNormalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2L)

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(100)
      .setNumHashTables(50)
      .setInputCol("normFeatures")
      .setOutputCol("hashes")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        useEmbeddings,
        sentenceFinisher,
        explodeVectors,
        vectorNormalizer,
        brp
      ))

    val pipelineModel = pipeline.fit(smallCorpus)
    val pipelineDF = pipelineModel.transform(smallCorpus)
      .withColumn("id", monotonically_increasing_id)

    pipelineDF.show()
    pipelineDF.select("features").show()

    pipelineDF.select("id", "text").show(false)

    val brpModel = pipelineModel.stages.last.asInstanceOf[BucketedRandomProjectionLSHModel]
    brpModel.approxSimilarityJoin(
      pipelineDF.select("normFeatures", "hashes", "id"),
      pipelineDF.select("normFeatures", "hashes", "id"),
      1.0,
      "EuclideanDistance")
      .select(
        $"datasetA.id".alias("idA"),
        $"datasetB.id".alias("idB"),
        $"EuclideanDistance")
      .filter("idA != idB") // not interested in self evaluation!
      .orderBy($"EuclideanDistance".asc)
      .show()
  }

  "UniversalSentenceEncoder" should "not fail on empty inputs" taggedAs SlowTest in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. . . . .... ..."),
      (3, "")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        useEmbeddings
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("sentence.result").show(false)
    pipelineDF.select("sentence_embeddings.result").show(false)
    pipelineDF.show()

  }


  "UniversalSentenceEncoder" should "correctly calculate sentence embeddings for multi-lingual" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings_use.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained("tfhub_use_multi", "xx")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
      .setLoadSP(true)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        useEmbeddings
      ))

    val pipelineModel = pipeline.fit(smallCorpus)
    val pipelineDF = pipelineModel.transform(smallCorpus)
    println(pipelineDF.count())
    pipelineDF.show

    pipelineModel.stages.last.asInstanceOf[UniversalSentenceEncoder].write.overwrite().save("./tmp_tfhub_use_multi")
    UniversalSentenceEncoder.load("./tmp_tfhub_use_multi")

  }

}
