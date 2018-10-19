package com.johnsnowlabs.nlp.annotators.parser.util

import com.johnsnowlabs.nlp.annotators.parser.util.DictionaryTypes.DictionaryTypes
import gnu.trove.map.hash.TIntIntHashMap

class DictionarySet(dictionaries: Array[Dictionary]) {

  private var counters: Array[TIntIntHashMap] = _
  private var isCounting = false

  def this () {
    this{
      val indexDictionaryTypes = DictionaryTypes.TYPE_END.id
      val dictionary = new Dictionary()
      Array.fill[Dictionary](indexDictionaryTypes)(dictionary)
    }
  }

  def lookupIndex(tag: DictionaryTypes, item: String): Int = {
    var id = this.dictionaries(tag.id).lookupIndex(item)

    if (this.isCounting && id > 0) {
      counters(tag.id).putIfAbsent(id, 0)
      counters(tag.id).increment(id)
    }

    id = if (id <= 0) 1 else id
    id
  }

  def getDictionarySize(tag: DictionaryTypes): Int = {
    val indexTag = tag.id
    this.dictionaries(indexTag).dictionarySize
  }

  def stopGrowth(tag: DictionaryTypes): Unit = {
    this.dictionaries(tag.id).stopGrowth()
  }

  def getDictionary(tag: DictionaryTypes): Dictionary = this.dictionaries(tag.id)

  def setCounters(): Unit = {
    isCounting = true
    counters = Array.fill[TIntIntHashMap](dictionaries.length)(new TIntIntHashMap)
  }

  def closeCounters(): Unit = {
    isCounting = false
    counters = null
  }

}
