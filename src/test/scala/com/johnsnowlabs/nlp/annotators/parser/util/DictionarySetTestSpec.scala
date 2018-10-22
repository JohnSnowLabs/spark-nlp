package com.johnsnowlabs.nlp.annotators.parser.util

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DictionarySet, DictionaryTypes}
import org.scalatest.FlatSpec

class DictionarySetTestSpec extends FlatSpec{

  "DictionarySet constructor" should "set attributes" in {
    val dictionarySet = new DictionarySet()

    assert(dictionarySet.isInstanceOf[DictionarySet])
  }

  "getDictionarySize method" should "return size when a valid tag is send" in {
    val dictionarySet = new DictionarySet()
    val expectedSize = 0
    val tag = DictionaryTypes.POS

    val dictionarySize = dictionarySet.getDictionarySize(tag)

    assert(dictionarySize == expectedSize)
  }

  "setCounters method" should "fill counters array" in {

    val dictionarySet = new DictionarySet()

    dictionarySet.setCounters()

  }

}
