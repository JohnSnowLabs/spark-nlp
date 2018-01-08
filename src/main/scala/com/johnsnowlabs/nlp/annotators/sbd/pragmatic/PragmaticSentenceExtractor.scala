package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence

/**
  * Reads through symbolized data, and computes the bounds based on regex rules following symbol meaning
 *
  * @param text symbolized text
  */
class PragmaticSentenceExtractor(text: String) {

  private val recoverySymbols = ("([" + PragmaticSymbols.symbolRecovery.keys.mkString + "])").r

  /** Goes through all sentences to store length and bounds of sentences */
  private def buildSentenceProperties(rawSentences: Array[String]) = {
    val sentences: Array[Sentence] = Array.ofDim[Sentence](rawSentences.length)
    var lastCharPosition = 0
    var i = 0
    while (i < sentences.length) {
      val rawSentence = rawSentences(i)
      val sentence = rawSentence.trim()
      val startPad = rawSentence.indexOf(sentence)

      val sentenceLastCharPos = lastCharPosition + rawSentence.length - 1
      sentences(i) = Sentence(
        sentence,
        lastCharPosition + startPad,
        lastCharPosition + startPad + sentence.length() - 1
      )
      lastCharPosition = sentenceLastCharPos + 1
      i = i + 1
    }
    sentences
  }

  /**
    * 1. Splits the text by boundary symbol that are not within a protection marker
    * 2. replaces all ignore marker symbols with nothing
    * 3. Clean all sentences that ended up being empty between boundaries
    * 4. Puts back all replacement symbols with their original meaning
    * 5. Collects sentence information
    * @return final sentence structure
    */
  def pull: Array[Sentence] = {
    val splitSentences = text
      .split(PragmaticSymbols.UNPROTECTED_BREAK_INDICATOR)
      .map(_.replaceAll(PragmaticSymbols.BREAK_INDICATOR, ""))
      .map(s => recoverySymbols.replaceAllIn(
        s, m => PragmaticSymbols.symbolRecovery
          .getOrElse(m.matched, throw new IllegalArgumentException("Invalid symbol in sentence recovery"))
      ))

    buildSentenceProperties(splitSentences)
      .filter(_.content.trim.nonEmpty)
  }
}
