package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproachDistributed, PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec

trait ChunkerBehaviors { this:FlatSpec =>

  case class SentenceParams(sentence: String, POSFormatSentence: Array[String],
                            regexParser: Array[String], correctChunkPhrases: Array[String])

  def testChunkingWithTrainedPOS(document: Array[String]): Unit = {
    it should "successfully transform a trained POS tagger annotator" in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(document).toDF("text")
      )
      val tokenized = AnnotatorBuilder.withTokenizer(data, sbd = false)

      val trainedTagger: PerceptronModel =
        new PerceptronApproach()
          .setInputCols("document", "token")
          .setOutputCol("pos")
          .setNIterations(3)
          .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
            ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
          .fit(tokenized)

      val POSdataset = trainedTagger.transform(tokenized)

      val chunker = new Chunker()
        .setInputCols(Array("pos"))
        .setOutputCol("chunk")
        .setRegexParsers(Array("(?:<JJ|DT>)(?:<NN|VBG>)+"))
        .transform(POSdataset)

      assert(chunker.select("chunk").count() > 0)

    }
  }

  def testChunkingWithTrainedPOSUsingPipeline(document: Array[String]): Unit = {
    it should "successfully transform POS tagger annotator with pipeline" in {
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
        .setRegexParsers(Array("(<NN>)+"))

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
      assert(transform.select("finished_chunk").count() > 0)
    }
  }

  def chunkerModelBuilder(dataset: Dataset[Row], regexParser: Array[String]): PipelineModel = {
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
      .setRegexParsers(regexParser)

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

  def testUserInputPOSTags(sentence: Array[String], tags: Array[Array[String]], regexParser: Array[String]): Unit = {
    it should "successfully generate chunk phrases from a regex parser" in {
      import SparkAccessor.spark.implicits._
      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(sentence.zip(tags)).toDF("text", "tags")
      )

      val model = this.chunkerModelBuilder(data, regexParser)
      val transform = model.transform(data)
      //transform.show(false)
      assert(transform.select("finished_chunks").count() > 0)
    }
  }

  def testRegexDoesNotMatch(sentence: Array[String], tags: Array[Array[String]], regexParser: Array[String]): Unit = {
    it should "successfully generate empty chunk phrases from a regex parser" in {
      import SparkAccessor.spark.implicits._

      val data = AnnotatorBuilder.withDocumentAssembler(
        SparkAccessor.spark.sparkContext.parallelize(sentence.zip(tags)).toDF("text", "tags")
      )

      val model = this.chunkerModelBuilder(data, regexParser)
      val finished_chunks = model.transform(data).select("finished_chunks").collect().map(row =>
        row.get(0).asInstanceOf[Seq[String]].toList)
      finished_chunks.map( row => assert(row.isEmpty))
    }
  }

  def testChunksGeneration(phrases: List[SentenceParams]): Unit = {
    it should "successfully generate correct chunks phrases for each sentence" in {
      import SparkAccessor.spark.implicits._
      import org.apache.spark.sql.functions._

      for (phrase <- phrases){

        val dataSeq = Seq((phrase.sentence, phrase.POSFormatSentence, phrase.correctChunkPhrases))
        val data = SparkAccessor.spark.sparkContext.parallelize(dataSeq).toDF("text", "tags", "correct_chunks")

        val model = this.chunkerModelBuilder(data, phrase.regexParser)
        var transform = model.transform(data)
        //transform.show(false)

        transform = transform.withColumn("equal_chunks",
          col("correct_chunks") === col("finished_chunks"))

        val equalChunks = transform.select("equal_chunks").collect()
        for (equalChunk <- equalChunks){
          assert(equalChunk.getBoolean(0))
        }

      }

    }
  }

  def testMultipleRegEx(dataset: Dataset[Row], regexParser: Array[String]): Unit = {
    it should "successfully generate chunks phrases" in {
      val model = this.chunkerModelBuilder(dataset, regexParser)
      val finished_chunks = model.transform(dataset).select("finished_chunks").collect().map( row =>
        row.get(0).asInstanceOf[Seq[String]].toList
      )
      finished_chunks.map( row => assert(row.nonEmpty))

    }
  }

}
