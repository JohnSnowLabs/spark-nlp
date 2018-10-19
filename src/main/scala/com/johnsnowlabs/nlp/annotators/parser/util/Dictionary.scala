package com.johnsnowlabs.nlp.annotators.parser.util

import gnu.trove.map.hash.TObjectIntHashMap

class Dictionary(capacity: Int) {

  private var mapDictionary: TObjectIntHashMap[String] = _
  private var numEntries = 0
  private var growthStopped = false

  def this() {
    this(10000)
    this.mapDictionary = new TObjectIntHashMap[String](capacity)
  }

  /** Return -1 (in old trove version) or 0 (in trove current verion) if entry isn't present. */
  def lookupIndex(entry: String): Int = {
    if (entry == null) throw new IllegalArgumentException("Can't lookup \"null\" in an Alphabet.")
    var ret = mapDictionary.get(entry)
    if (ret <= 0 && !growthStopped) {
      numEntries += 1
      ret = numEntries
      mapDictionary.put(entry, ret)
    }
    ret
  }

  def toArray: Array[AnyRef] = this.mapDictionary.keys

  def dictionarySize: Int = numEntries

  //private[utils] def stopGrowth(): Unit = {
  def stopGrowth(): Unit = {
    growthStopped = true
  }

}
