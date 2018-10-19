package com.johnsnowlabs.nlp.annotators.parser.feature

import com.johnsnowlabs.nlp.annotators.parser.util.Alphabet
import gnu.trove.set.hash.TLongHashSet

class SyntacticFeatureFactory(wordAlphabet: Alphabet, stoppedGrowth: Boolean, featureHashSet: TLongHashSet,
                              numberWordFeatures: Int, numberLabeledArcFeatures: Int) {

  private val BITS = 30

  private var tokenStart = 1
  private var tokenEnd = 2
  private var tokenMid = 3

  def setTokenStart(tokenStart: Int): Unit = {
    this.tokenStart = tokenStart
  }

  def setTokenEnd(tokenEnd: Int): Unit = {
    this.tokenEnd = tokenEnd
  }

  def setTokenMid(tokenMid: Int): Unit = {
    this.tokenMid = tokenMid
  }

  private var tagNumBits = 0
  private var wordNumBits = 0
  private var depNumBits = 0
  private var flagBits = 0

  def getTagNumBits: Int = tagNumBits

  def setTagNumBits(tagNumBits: Int): Unit = {
    this.tagNumBits = tagNumBits
  }

  def getWordNumBits: Int = wordNumBits

  def setWordNumBits(wordNumBits: Int): Unit = {
    this.wordNumBits = wordNumBits
  }

  def getDepNumBits: Int = depNumBits

  def setDepNumBits(depNumBits: Int): Unit = {
    this.depNumBits = depNumBits
  }

  def getFlagBits: Int = flagBits

  def setFlagBits(flagBits: Int): Unit = {
    this.flagBits = flagBits
  }

  def this() {
    this(new Alphabet(), false, new TLongHashSet(100000), 0, ((1L << (30 - 2)) - 1).toInt)
  }

}
