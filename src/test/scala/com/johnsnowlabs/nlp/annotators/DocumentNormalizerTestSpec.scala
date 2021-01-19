package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.{FastTest, SlowTest}
import com.johnsnowlabs.nlp.AnnotatorType
import org.scalatest.FlatSpec


class DocumentNormalizerTestSpec extends FlatSpec with DocumentNormalizerBehaviors {
  val documentNormalizer = new DocumentNormalizer()

  "a DocumentNormalizer output" should s"be of type ${AnnotatorType.DOCUMENT}" in {
    assert(documentNormalizer.outputAnnotatorType == AnnotatorType.DOCUMENT)
  }

  it should "print correctly slow" taggedAs SlowTest in {
    println("CIAO Slow")
  }

  it should "print correctly fast" taggedAs FastTest in {
    println("CIAO FAST")
  }
}
