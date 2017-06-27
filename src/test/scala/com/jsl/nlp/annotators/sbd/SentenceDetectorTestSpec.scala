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
    override def prepare: SBDApproach = this
    override def extract: Array[Sentence] = Array(Sentence("A dummy sentence", 0, 20))
    override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = ???
  }

  val sentenceDetector = new SentenceDetector
  sentenceDetector.setModel(new DummyApproach)

  "a SentenceDetector" should s"be of type ${SentenceDetector.aType}" taggedAs Tag("LinuxOnly") in {
    assert(sentenceDetector.annotatorType == SentenceDetector.aType, "because types are not properly set up")
  }

}
