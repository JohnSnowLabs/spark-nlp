package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.annotators.common.WritableAnnotatorComponent
import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent
import org.scalatest._

/**
  * Created by Saif Addin on 6/1/2017.
  */
class SentenceDetectorTestSpec extends FlatSpec {

  private class DummyApproach extends SBDApproach {
    override val description = "dummy description"
    override def extractBounds(text: String): Array[Sentence] = Array(Sentence("A dummy sentence", 0, 20))
    override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = ???
  }

  val sentenceDetector = new SentenceDetector
  sentenceDetector.setModel(new DummyApproach)

  "a SentenceDetector" should s"be of type ${SentenceDetector.annotatorType}" taggedAs Tag("LinuxOnly") in {
    assert(sentenceDetector.annotatorType == SentenceDetector.annotatorType, "because types are not properly set up")
  }

}
