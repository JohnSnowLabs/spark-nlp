package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.common.{WritableAnnotatorComponent, TaggedSentence, TokenizedSentence}
import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent
import org.scalatest._

/**
  * Created by Saif Addin on 6/1/2017.
  */

class POSTaggerTestSpec extends FlatSpec {

  class DummyApproach extends POSApproach {
    class DummyModel extends POSModel[List[(String, Int)]] {
      override def predict(features: List[(String, Int)]): String = "dummyPrediction"
    }
    override val description = "dummy description"
    override val model = new DummyModel

    override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = ???
    override def tag(sentences: Array[TokenizedSentence]): Array[TaggedSentence] = Array()
  }

  val posTagger = new POSTagger
  posTagger.setModel(new DummyApproach)

  "a SentenceDetector" should s"be of type ${POSTagger.annotatorType}" taggedAs Tag("LinuxOnly") in {
    assert(posTagger.annotatorType == POSTagger.annotatorType, "because types are not properly set up")
  }

}
