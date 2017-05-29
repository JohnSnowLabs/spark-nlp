package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.sbd.Sentence

/**
  * Created by Saif Addin on 5/6/2017.
  */
class PragmaticSentenceExtractor(text: String) {

  private val recoverySymbols = ("([" + PragmaticSymbols.sentenceRecovery.keys.mkString + "])").r

  /**
    * Goes through all sentences and records substring beginning and end
    * May think a more functional way? perhaps a foldRight with vector (i,lastChar)?
    * @return
    */
  private def buildSentenceProperties(rawSentences: Array[String]) = {
    val sentences: Array[Sentence] = Array.ofDim[Sentence](rawSentences.length)
    var lastCharPosition = 0
    var i = 0
    while (i < sentences.length) {
      val sentenceContent = rawSentences(i)
      val sentenceLastCharPos = sentenceContent.length - 1
      sentences(i) = Sentence(
        sentenceContent,
        lastCharPosition,
        sentenceLastCharPos
      )
      lastCharPosition = sentenceLastCharPos
      i = i + 1
    }
    sentences
  }


  def pull: Array[Sentence] = {
    val splitSentences: Array[String] = text
      // Split by breakers ignoring breaks within protection
      .split(PragmaticSymbols.PROTECTED_BREAKER)
      // clean ignored breakers
      .map(_.replaceAll(PragmaticSymbols.BREAK_INDICATOR, ""))
      // leave only useful content
      .map(_.trim).filter(_.nonEmpty)
    val rawSentences: Array[String] = splitSentences.map(s => recoverySymbols.replaceAllIn(
      s, m => PragmaticSymbols.sentenceRecovery
        .getOrElse(m.matched, throw new IllegalArgumentException("Invalid symbol in sentence recovery"))))
    buildSentenceProperties(rawSentences)
  }

}
