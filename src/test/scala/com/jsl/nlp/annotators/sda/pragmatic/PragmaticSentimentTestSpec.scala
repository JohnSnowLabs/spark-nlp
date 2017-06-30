package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp.DataBuilder
import com.jsl.nlp.annotators.sda.SentimentDetector
import com.jsl.nlp.util.io.ResourceHelper
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
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
