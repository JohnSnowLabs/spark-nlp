package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp.DataBuilder
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
class PragmaticSentimentTestSpec extends FlatSpec with PragmaticSentimentBehaviors {

  val sentimentSentence1 = "The staff of the restaurant is nice and the eggplant is bad".split(" ")
  val sentimentSentence2 = "I recommend others to avoid".split(" ")
  val sentimentSentences = Array(
    TaggedSentence(
      sentimentSentence1.map(TaggedWord(_, "?NOTAG?"))
    ),
    TaggedSentence(
      sentimentSentence2.map(TaggedWord(_, "?NOTAG?"))
    )
  )

  "an isolated sentiment detector" should behave like isolatedSentimentDetector(sentimentSentences, 1.0)

  "a spark based sentiment detector" should behave like sparkBasedSentimentDetector(
    DataBuilder.basicDataBuild("The staff of the restaurant is nice and the eggplant is bad." +
      " I recommend others to avoid.")
  )

}
