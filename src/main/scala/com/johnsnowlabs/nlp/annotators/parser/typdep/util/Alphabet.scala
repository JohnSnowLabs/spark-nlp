package com.johnsnowlabs.nlp.annotators.parser.typdep.util

import gnu.trove.map.hash.TLongIntHashMap

class Alphabet(capacity: Int) {

  var mapAlphabet: TLongIntHashMap = _
  private var numEntries = 0
  private var growthStopped = false

  def this() {
    this(10000)
    this.mapAlphabet = new TLongIntHashMap(capacity)
  }

  /** Return -1 if entry isn't present. */
  def lookupIndex(entry: Long, value: Int): Int = {
    var ret = mapAlphabet.get(entry)
    if (ret <= 0 && !growthStopped) {
      numEntries += 1
      ret = value + 1
      mapAlphabet.put(entry, ret)
    }
    ret - 1 // feature id should be 0-based
  }

  /** Return -1 if entry isn't present. */
  def lookupIndex(entry: Long): Int = {
    var ret = mapAlphabet.get(entry)
    if (ret <= 0 && !growthStopped) {
      numEntries += 1
      ret = numEntries
      mapAlphabet.put(entry, ret)
    }
    ret - 1 // feature id should be 0-based

  }

  def stopGrowth(): Unit = {
    growthStopped = true
  }

}
