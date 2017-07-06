package com.jsl.nlp.annotators.sda
import com.jsl.nlp.annotators.common.TaggedSentence
import com.jsl.nlp.annotators.param.{SerializedAnnotatorComponent, WritableAnnotatorComponent}
import org.scalatest._

/**
  * Created by Saif Addin on 6/18/2017.
  */
class SentimentDetectorTestSpec extends FlatSpec {
  private class DummyApproach extends SentimentApproach {
    override val description = "dummy description"
    override val requiresLemmas: Boolean = false
    override val requiresPOS: Boolean = false

    override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = ???
    override def score(taggedSentences: Array[TaggedSentence]): Double = 1.0
  }

  val sentimentDetector = new SentimentDetector()
  sentimentDetector.setModel(new DummyApproach)

  "a SentimentDetector" should s"be of type ${sentimentDetector.annotatorType}" taggedAs Tag("LinuxOnly") in {
    assert(sentimentDetector.annotatorType == SentimentDetector.annotatorType, "because types are not properly set up")
  }

}
