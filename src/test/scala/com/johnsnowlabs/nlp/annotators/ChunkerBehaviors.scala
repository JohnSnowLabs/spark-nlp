package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.annotators.ChunkerBehaviors

trait ChunkerBehaviors { this:FlatSpec =>

  case class Phrases(sentence: String, POSFormatSentence: Array[String],
                     regexParser: String, correctChunkPhrases: Array[String])

  def testPOSForChunking(document: Array[String]): Unit = {
    it should "successfully transform a trained POS tag" in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(document).toDF("text")
      )
      val tokenized = AnnotatorBuilder.withTokenizer(data, sbd = false)
      tokenized.show(false)

      val trainedTagger: PerceptronModel =
        new PerceptronApproach()
          .setInputCols("document", "token")
          .setOutputCol("pos")
          .setNIterations(3)
          .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
            ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
          .fit(tokenized)

      trainedTagger.transform(tokenized).show(false)

      val finisher = new Finisher()
        .setInputCols("normalized")
        .setOutputAsArray(false)

    }
  }

  def testChunkingWithTrainedPOS(document: Array[String]): Unit = {
    it should "successfully transform a trained POS tag with pipeline" in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(document).toDF("text")
      )
      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val POSTag = new PerceptronApproach()
        .setInputCols("document", "token")
        .setOutputCol("pos")
        .setNIterations(3)
        .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
          ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))

      val chunker = new Chunker()
        .setInputCols(Array("pos"))
        .setOutputCol("chunk")
        .setRegexParser("(<NN>)+")

      val finisher = new Finisher()
        .setInputCols("chunk")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          POSTag,
          chunker,
          finisher
        ))

      val model = pipeline.fit(data)
      val transform = model.transform(data)
      transform.show(false)
    }
  }

  def chunkModelBuilder(dataset: Dataset[Row], regexParser: String): PipelineModel = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val manualTrainedPos = new PerceptronApproach()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setPosColumn("tags")

    val chunker = new Chunker()
      .setInputCols(Array("pos"))
      .setOutputCol("chunks")
      .setRegexParser(regexParser)

    val finisher = new Finisher()
      .setInputCols("chunks")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        manualTrainedPos,
        chunker,
        finisher
      ))

    val model = pipeline.fit(dataset)
    model
  }

  def testUserInputPOSTags(document: Array[String], tags: Array[Array[String]], regexParser: String): Unit = {
    it should "successfully train from a POS Column" in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(document.zip(tags)).toDF("text", "tags")
      )

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val manualTrainedPos = new PerceptronApproach()
        .setInputCols("document", "token")
        .setOutputCol("pos")
        .setPosColumn("tags")

      val chunker = new Chunker()
        .setInputCols(Array("pos"))
        .setOutputCol("chunks")
        .setRegexParser(regexParser)
      //.setRegexParser("<DT>?<JJ>*<NN>")
        //.setRegexParser("(<DT>)?(<JJ>)*(<NN>)")

      val finisher = new Finisher()
        .setInputCols("chunks")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          manualTrainedPos,
          chunker,
          finisher
        ))

      val model = pipeline.fit(data)
      val transform = model.transform(data)
      transform.show(false)

    }
  }

  def testSeveralSentences(phrases: List[Phrases]): Unit = {
    it should "successfully generate chunks from regex parser" in {
      import SparkAccessor.spark.implicits._
      import org.apache.spark.sql.functions._

      for (phrase <- phrases){

        val dataSeq = Seq((phrase.sentence, phrase.POSFormatSentence, phrase.correctChunkPhrases))
        val data = SparkAccessor.spark.sparkContext.parallelize(dataSeq).toDF("text", "tags", "correct_chunks")
        //data.show(false)

        val model = this.chunkModelBuilder(data, phrase.regexParser)
        var transform = model.transform(data)
        //transform.show(false)

        transform = transform.withColumn("equal_chunks",
          col("correct_chunks") === col("finished_chunks"))
        //transform.show(false)

        //https://medium.com/@mrpowers/testing-spark-applications-8c590d3215fa
        val equalChunks = transform.select("equal_chunks").collect()
        for (equalChunk <- equalChunks){
          assert(equalChunk.getBoolean(0))
        }

      }

    }
  }

  def testDocument(dataset: Dataset[Row], regexParser: String): Unit = {
    it should "successfully generate chunks from regex parser" in {
      val model = this.chunkModelBuilder(dataset, regexParser)
      model.transform(dataset).show(false)

    }
  }

}
