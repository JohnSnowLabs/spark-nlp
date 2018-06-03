package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

trait ChunkerBehaviors { this:FlatSpec =>

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

  def testPOSForChunkingWithPipeline(document: Array[String]): Unit = {
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
        .setInputCols(Array("pos", "document"))
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

  def testUserInputPOSTags(document: Array[String], tags: Array[Array[String]]): Unit = {
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
        .setOutputCol("chunk")
        .setRegexParser("<DT>?<JJ>*<NN>")
        //.setRegexParser("(<DT>)?(<JJ>)*(<NN>)")

      val finisher = new Finisher()
        .setInputCols("chunk")

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

}
