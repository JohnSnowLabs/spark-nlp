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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.tags.FastTest

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}

import org.scalatest.flatspec.AnyFlatSpec

trait ChunkerBehaviors {
  this: AnyFlatSpec =>

  case class SentenceParams(
      sentence: String,
      POSFormatSentence: Array[String],
      regexParser: Array[String],
      correctChunkPhrases: Array[String])

  def testChunkingWithTrainedPOS(document: Array[String]): Unit = {
    it should "successfully transform a trained POS tagger annotator" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withFullPragmaticSentenceDetector(
        AnnotatorBuilder.withDocumentAssembler(
          SparkAccessor.spark.sparkContext.parallelize(document).toDF("text")))

      val trainingPerceptronDF = POS().readDataset(
        ResourceHelper.spark,
        "src/test/resources/anc-pos-corpus-small/",
        "|",
        "tags")
      val tokenized = AnnotatorBuilder.withTokenizer(data, sbd = false)

      val trainedTagger: PerceptronModel =
        new PerceptronApproach()
          .setInputCols("sentence", "token")
          .setOutputCol("pos")
          .setPosColumn("tags")
          .setNIterations(3)
          .fit(trainingPerceptronDF)

      val POSdataset = trainedTagger.transform(tokenized)

      val chunker = new Chunker()
        .setInputCols(Array("sentence", "pos"))
        .setOutputCol("chunk")
        .setRegexParsers(Array("(?:<JJ|DT>)(?:<NN|VBG>)+"))
        .transform(POSdataset)

      assert(chunker.select("chunk").count() > 0)

    }
  }

  def testChunkingWithTrainedPOSUsingPipeline(document: Array[String]): Unit = {
    it should "successfully transform POS tagger annotator with pipeline" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(document).toDF("text"))

      val trainingPerceptronDF = POS().readDataset(
        ResourceHelper.spark,
        "src/test/resources/anc-pos-corpus-small/",
        "|",
        "tags")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val sentence = new SentenceDetector()
        .setInputCols("document")
        .setOutputCol("sentence")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("sentence"))
        .setOutputCol("token")

      val POSTag = new PerceptronApproach()
        .setInputCols("sentence", "token")
        .setOutputCol("pos")
        .setPosColumn("tags")
        .setNIterations(3)

      val chunker = new Chunker()
        .setInputCols(Array("sentence", "pos"))
        .setOutputCol("chunk")
        .setRegexParsers(Array("(<NN>)+"))

      val finisher = new Finisher()
        .setInputCols("chunk")

      val pipeline = new Pipeline()
        .setStages(Array(documentAssembler, sentence, tokenizer, POSTag, chunker, finisher))

      val model = pipeline.fit(trainingPerceptronDF)
      val transform = model.transform(data)
      assert(transform.select("finished_chunk").count() > 0)
    }
  }

  def chunkerModelBuilder(dataset: Dataset[Row], regexParser: Array[String]): PipelineModel = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val manualTrainedPos = new PerceptronApproach()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")
      .setPosColumn("tags")

    val chunker = new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunks")
      .setRegexParsers(regexParser)

    val finisher = new Finisher()
      .setInputCols("chunks")

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentence, tokenizer, manualTrainedPos, chunker, finisher))

    val model = pipeline.fit(dataset)
    model
  }

  def testUserInputPOSTags(
      sentence: Array[String],
      path: String,
      regexParser: Array[String]): Unit = {
    it should "successfully generate chunk phrases from a regex parser" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._
      val testData = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(sentence).toDF("text"))

      val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, path, "|", "tags")

      val model = this.chunkerModelBuilder(trainingPerceptronDF, regexParser)
      val transform = model.transform(testData)
      // transform.show(false)
      assert(transform.select("finished_chunks").count() > 0)
    }
  }

  def testRegexDoesNotMatch(
      sentence: Array[String],
      path: String,
      regexParser: Array[String]): Unit = {
    it should "successfully generate empty chunk phrases from a regex parser" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._

      val testData = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(sentence).toDF("text"))

      val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, path, "|", "tags")

      val model = this.chunkerModelBuilder(trainingPerceptronDF, regexParser)
      val finished_chunks = model
        .transform(testData)
        .select("finished_chunks")
        .collect()
        .map(row => row.get(0).asInstanceOf[Seq[String]].toList)
      finished_chunks.map(row => assert(row.isEmpty))
    }
  }

  def testChunksGeneration(phrases: List[SentenceParams]): Unit = {
    it should "successfully generate correct chunks phrases for each sentence" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._
      import org.apache.spark.sql.functions._

      for (phrase <- phrases) {

        val tokensLabelsDF = Seq((phrase.sentence, phrase.POSFormatSentence))
          .toDF("sentence", "labels")
          .withColumn("tokens", split($"sentence", " "))

        val trainingPerceptronDF = POS().readFromDataframe(tokensLabelsDF)

        val dataSeq = Seq((phrase.sentence, phrase.POSFormatSentence, phrase.correctChunkPhrases))
        val data = SparkAccessor.spark.sparkContext
          .parallelize(dataSeq)
          .toDF("text", "tags", "correct_chunks")

        val model = this.chunkerModelBuilder(trainingPerceptronDF, phrase.regexParser)

        var transform = model.transform(data)
        // transform.show(false)

        transform =
          transform.withColumn("equal_chunks", col("correct_chunks") === col("finished_chunks"))

        val equalChunks = transform.select("equal_chunks").collect()
        for (equalChunk <- equalChunks) {
          assert(equalChunk.getBoolean(0))
        }

      }

    }
  }

  def testMultipleRegEx(dataset: Dataset[Row], regexParser: Array[String]): Unit = {
    it should "successfully generate chunks phrases" taggedAs FastTest in {
      import SparkAccessor.spark.implicits._
      import org.apache.spark.sql.functions.split

      val tokensLabelsDF = dataset
        .withColumn("tokens", split($"text", " "))

      val trainingPerceptronDF = POS().readFromDataframe(tokensLabelsDF)

      val model = this.chunkerModelBuilder(trainingPerceptronDF, regexParser)

      val finished_chunks = model
        .transform(dataset)
        .select("finished_chunks")
        .collect()
        .map(row => row.get(0).asInstanceOf[Seq[String]].toList)
      finished_chunks.map(row => assert(row.nonEmpty))

    }
  }

}
