package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.sbd.Sentence

/**
  * Created by Saif Addin on 5/6/2017.
  */
class PragmaticSentenceExtractor(text: String) {

  private val breakMatcher = ("[" + PragmaticSymbols.getSentenceBreakers.mkString + "]").r
  private val nonBreakMatcher = ("[" + PragmaticSymbols.getSentenceNonBreakers.mkString + "]").r

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
    val splitReadyText = breakMatcher.replaceAllIn(text, PragmaticSymbols.FINAL_BREAK)
    val splitSentences: Array[String] = text.split(splitReadyText)
    val rawSentences: Array[String] = splitSentences.map(
      rawSentence => nonBreakMatcher.replaceAllIn(rawSentence, PragmaticSymbols.FINAL_NON_BREAK)
    )
    buildSentenceProperties(rawSentences)
  }

}
