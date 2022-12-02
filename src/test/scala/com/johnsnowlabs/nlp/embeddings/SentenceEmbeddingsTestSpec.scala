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

import com.johnsnowlabs.nlp.annotators.classifier.dl.{ClassifierDLApproach, ClassifierDLModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorBuilder, EmbeddingsFinisher, Finisher}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest.flatspec.AnyFlatSpec

class SentenceEmbeddingsTestSpec extends AnyFlatSpec {

  "SentenceEmbeddings" should "correctly calculate sentence embeddings in WordEmbeddings" taggedAs FastTest in {

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
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_sentence_embeddings")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentence,
          tokenizer,
          embeddings,
          embeddingsSentence,
          sentenceFinisher))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.select("embeddings.metadata").show(2)
    pipelineDF.select("embeddings.embeddings").show(2)
    pipelineDF.select("embeddings.result").show(2)

    pipelineDF.select("sentence_embeddings").show(2)
    pipelineDF.select("sentence_embeddings.embeddings").show(1)
    pipelineDF
      .select(size(pipelineDF("sentence_embeddings.embeddings")).as("sentence_embeddings_size"))
      .show

    pipelineDF.select("finished_sentence_embeddings").show(1)
    pipelineDF
      .select(size(pipelineDF("finished_sentence_embeddings")).as("sentence_embeddings_size"))
      .show

  }

  // too large for Travis
  "SentenceEmbeddings" should "correctly calculate sentence embeddings in BertEmbeddings" taggedAs SlowTest in {

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
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")

    val finisher = new Finisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_embeddings")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentence, tokenizer, embeddings, embeddingsSentence, finisher))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.select("embeddings.metadata").show(1)
    pipelineDF.select("embeddings.embeddings").show(1)
    pipelineDF.select("embeddings.result").show(1)

    pipelineDF.select("sentence_embeddings").show(1)
    pipelineDF.select("sentence_embeddings.embeddings").show(1)
    pipelineDF
      .select(size(pipelineDF("sentence_embeddings.embeddings")).as("sentence_embeddings_size"))
      .show

    pipelineDF.select("finished_embeddings").show(1)
    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("sentence_embeddings_size")).show

  }

  "SentenceEmbeddings" should "not crash on empty embeddings" taggedAs FastTest in {

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
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(
        Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_sentence_embeddings")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentence,
          tokenizer,
          stopWordsCleaner,
          embeddings,
          embeddingsSentence,
          sentenceFinisher))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.show(2)
  }

  "SentenceEmbeddings" should "correctly pass storageRef down the pipeline" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/classifier/sentiment.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(1)
      .setLr(5e-3f)
      .setDropout(0.5f)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentence,
          tokenizer,
          embeddings,
          embeddingsSentence,
          docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
    val pipelineDF = pipelineModel.transform(smallCorpus)

    val embedStorageRef = embeddings.getStorageRef

    val sentEmbedRef = embeddingsSentence.getStorageRef
    val sentEmbedRefPipeModel =
      pipelineModel.stages(4).asInstanceOf[SentenceEmbeddings].getStorageRef
    val classifierStorageRef =
      pipelineModel.stages.last.asInstanceOf[ClassifierDLModel].getStorageRef

    assert(sentEmbedRef == embedStorageRef)
    assert(sentEmbedRefPipeModel == embedStorageRef)
    assert(classifierStorageRef == embedStorageRef)
  }
}
