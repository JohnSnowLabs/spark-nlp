package com.johnsnowlabs.nlp.annotators.parser.util

import org.scalatest.FlatSpec

class DictionarySetTestSpec extends FlatSpec{

  "DictionarySet constructor" should "set attributes" in {
    val expectedLength = 4
    val dictionarySet = new DictionarySet()

    assert(dictionarySet.isInstanceOf[DictionarySet])
    assert(dictionarySet.dictionaries.length == expectedLength)

  }

  "getDictionarySize method" should "return size when a valid tag is send" in {
    val dictionarySet = new DictionarySet()
    val expectedSize = 0
    val tag = DictionaryTypes.POS

    val dictionarySize = dictionarySet.getDictionarySize(tag)

    assert(dictionarySize == expectedSize)
  }

}
