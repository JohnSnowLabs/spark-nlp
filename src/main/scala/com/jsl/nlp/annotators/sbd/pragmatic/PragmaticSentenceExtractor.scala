package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.sbd.Sentence

/**
  * Created by Saif Addin on 5/6/2017.
  */
class PragmaticSentenceExtractor(text: String) {
  /*
  ** hardcode: alt + 197 == ┼
  */
  private val splitWord = "┼"
  private var wip: String = text

  /**
    * Goes through all sentences and records substring beginning and end
    * May think a more functional way? perhaps a foldRight with vector (i,lastChar)?
    * @return
    */
  def pull: Array[Sentence] = {
    val rawSentences: Array[String] = wip.split(splitWord)
    val sentences: Array[Sentence] = Array.ofDim[Sentence](rawSentences.length)
    var lastCharPosition = 0
    var i = 0
    while (i < rawSentences.length) {
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

}
