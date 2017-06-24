package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.annotators.common.AnnotatorApproach
import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach
import org.scalatest._

/**
  * Created by Saif Addin on 6/1/2017.
  */
class SentenceDetectorTestSpec extends FlatSpec {

  private class DummyApproach extends SBDApproach {
    override val description = "dummy description"
    override def prepare: SBDApproach = this
    override def extract: Array[Sentence] = Array(Sentence("A dummy sentence", 0, 20))
    override def serialize: SerializedAnnotatorApproach[_ <: AnnotatorApproach] = ???
  }

  val sentenceDetector = new SentenceDetector
  sentenceDetector.setModel(new DummyApproach)

  "a SentenceDetector" should s"be of type ${SentenceDetector.aType}" in {
    assert(sentenceDetector.aType == SentenceDetector.aType, "because types are not properly set up")
  }

}
