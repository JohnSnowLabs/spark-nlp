package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec


class DocumentNormalizerTestSpec extends FlatSpec with DocumentNormalizerBehaviors {
  val documentNormalizer = new DocumentNormalizer()

  "a DocumentNormalizer output" should s"be of type ${AnnotatorType.DOCUMENT}" taggedAs FastTest in {
    assert(documentNormalizer.outputAnnotatorType == AnnotatorType.DOCUMENT)
  }
}
