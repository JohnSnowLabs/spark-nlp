/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.feature.{Normalizer, SQLTransformer}
import org.scalatest._

class EmbeddingsFinisherTestSpec extends FlatSpec {

  "EmbeddingsFinisher" should "correctly transform embeddings into array of floats for Spark ML" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(false)

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings", "embeddings")
      .setOutputCols("finished_sentence_embeddings", "finished_embeddings")
      .setOutputAsVector(false)
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        embeddingsFinisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    /*
    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("sentence_embeddings_size")).show
    pipelineDF.select("finished_embeddings").show(2)


    pipelineDF.select(size(pipelineDF("finished_sentence_embeddings")).as("sentence_embeddings_size")).show
    pipelineDF.select("finished_sentence_embeddings").show(2)

    val explodedVectors = pipelineDF.select($"sentence", explode($"finished_sentence_embeddings").as("features"))

    explodedVectors.select("features").show

    pipelineDF.printSchema()
    explodedVectors.printSchema()
    */
  }

  "EmbeddingsFinisher" should "correctly transform embeddings into Vectors and normalize it by Spark ML" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(false)

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings", "embeddings")
      .setOutputCols("sentence_embeddings_vectors", "embeddings_vectors")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val explodeVectors = new SQLTransformer().setStatement(
      "SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")

    // Normalize each Vector using $L^1$ norm.
    val vectorNormalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        embeddingsFinisher,
        explodeVectors,
        vectorNormalizer
      ))

    val pipelineModel = pipeline.fit(smallCorpus)
    val pielineDF = pipelineModel.transform(smallCorpus)

    /*
    pielineDF.show()
    pielineDF.printSchema()

    pielineDF.select(size(pielineDF("embeddings_vectors")).as("sentence_embeddings_size")).show
    pielineDF.select("embeddings_vectors").show(2)

    pielineDF.select(size(pielineDF("sentence_embeddings_vectors")).as("sentence_embeddings_size")).show
    pielineDF.select("sentence_embeddings_vectors").show(2)

    pielineDF.select("features").show(2)

    pielineDF.select("normFeatures").show(2)
    */
  }

}
