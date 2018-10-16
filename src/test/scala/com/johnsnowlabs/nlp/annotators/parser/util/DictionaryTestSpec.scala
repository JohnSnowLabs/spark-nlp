package com.johnsnowlabs.nlp.annotators.parser.util

import org.scalatest.FlatSpec

class DictionaryTestSpec extends FlatSpec {

  "Dictionary constructor" should "set attributes" in {

    val dictionary = new Dictionary()

    assert(dictionary.toArray.isInstanceOf[Array[AnyRef]])
    assert(dictionary.dictionarySize == 0)

  }

}
