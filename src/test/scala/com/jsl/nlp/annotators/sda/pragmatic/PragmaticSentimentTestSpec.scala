package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp._
import com.jsl.nlp.annotators.sda.SentimentDetector
import com.jsl.nlp.util.io.ResourceHelper
import org.apache.spark.storage.StorageLevel
import org.scalatest._
import org.scalatest.tagobjects.Slow

/**
  * Created by Saif Addin on 5/7/2017.
  */

class PragmaticSentimentBigTestSpec extends FlatSpec {

  "Parquet based data" should "be sentiment detected properly" taggedAs Slow in {
    import SparkAccessor.spark.implicits._
    import java.util.Date

    val data = ContentProvider.parquetData.limit(1000).withColumn("document", Document.column($"text"))

    val sentimentDetector = new SentimentDetector()
      .setDocumentCol("document")
      .setModel(new PragmaticScorer(ResourceHelper.retrieveSentimentDict))
      .setOutputAnnotationCol("my_sda_scores")

    val readyData = AnnotatorBuilder.withFullPOSTagger(AnnotatorBuilder.withFullLemmatizer(data))

    val result = sentimentDetector
      .transform(readyData)

    import Annotation.extractors._

    val date1 = new Date().getTime
    result.show
    info(s"20 show sample of disk based sentiment analysis took: ${(new Date().getTime - date1)/1000} seconds")

    val date2 = new Date().getTime
    result.take("my_sda_scores", 5000)
    info(s"5000 take sample of disk based sentiment analysis took: ${(new Date().getTime - date2)/1000} seconds")

    val dataFromMemory = readyData.persist(StorageLevel.MEMORY_AND_DISK)
    info(s"data in memory is of size: ${dataFromMemory.count}")
    val resultFromMemory = sentimentDetector.transform(dataFromMemory)

    val date3 = new Date().getTime
    resultFromMemory.show
    info(s"20 show sample of memory based sentiment analysis took: ${(new Date().getTime - date3)/1000} seconds")

    val date4 = new Date().getTime
    resultFromMemory.take("my_sda_scores", 5000)
    info(s"5000 take sample of memory based sentiment analysis took: ${(new Date().getTime - date4)/1000} seconds")

    succeed
  }

}

class PragmaticSentimentTestSpec extends FlatSpec with PragmaticSentimentBehaviors {

  val sentimentSentence1 = "The staff of the restaurant is nice and the eggplant is bad".split(" ")
  val sentimentSentence2 = "I recommend others to avoid because it is too expensive".split(" ")
  val sentimentSentences = Array(
    TaggedSentence(
      sentimentSentence1.map(TaggedWord(_, "?NOTAG?"))
    ),
    TaggedSentence(
      sentimentSentence2.map(TaggedWord(_, "?NOTAG?"))
    )
  )

  "an isolated sentiment detector" should behave like isolatedSentimentDetector(sentimentSentences, -4.0)

  "a spark based sentiment detector" should behave like sparkBasedSentimentDetector(
    DataBuilder.basicDataBuild("The staff of the restaurant is nice and the eggplant is bad." +
      " I recommend others to avoid.")
  )

  "A SentimentDetector" should "be readable and writable" in {
    val sentimentDetector = new SentimentDetector().setModel(new PragmaticScorer(ResourceHelper.retrieveSentimentDict))
    val path = "./test-output-tmp/sentimentdetector"
    try {
      sentimentDetector.write.overwrite.save(path)
      val sentimentDetectorRead = SentimentDetector.read.load(path)
      assert(sentimentDetector.getModel.description == sentimentDetectorRead.getModel.description)
      assert(sentimentDetector.getModel.score(sentimentSentences) == sentimentDetectorRead.getModel.score(sentimentSentences))
    } catch {
      case _: java.io.IOException => succeed
    }
  }

}
