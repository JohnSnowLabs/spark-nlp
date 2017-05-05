package com.jsl.nlp.annotators.sbd

/**
  * Created by Saif Addin on 5/5/2017.
  */
class PragmaticDetection extends SBDApproach {

  override val description: String = "rule-based sentence detector. based off pragmatic sentence detector."

  override def prepare(text: String): RawContent = new PragmaticRawContent(text)

  override def clean(rawContent: RawContent): PragmaticReadyContent = {
    rawContent match {
      case pragmaticRawContent: PragmaticRawContent =>
        pragmaticRawContent
          .formatAlphabeticalLists
          .formatRomanNumeralLists
          .formatNumberedListWithPeriods
          .formatNumberedListsWithParens
          .cleanse
      case _ => throw new IllegalArgumentException("Pragmatic detector received raw content of undefined type")
    }

  }

  override def extract(readyContent: ReadyContent): Array[Sentence] = {
    readyContent match {
      case pragmaticReadyContent: PragmaticReadyContent =>
        pragmaticReadyContent
          .replaceAbbreviations
          .replaceAbbreviations
          .replaceContinuousPunctuation
          .pull
    }
  }

}

class PragmaticRawContent(text: String)(implicit val approach: SBDApproach = this) extends RawContent {
  private var wip: String = text
  /**
    * The following functions should alter the value of 'wip'
    * since cleanse will return the final version
    * @return
    */
  def formatAlphabeticalLists: this.type = ???
  def formatRomanNumeralLists: this.type = ???
  def formatNumberedListWithPeriods: this.type = ???
  def formatNumberedListsWithParens: this.type = ???

  def cleanse: PragmaticReadyContent = new PragmaticReadyContent(wip)
}

class PragmaticReadyContent(text: String)(implicit val approach: SBDApproach = this) extends ReadyContent {
  /*
  ** hardcode: alt + 197 == ┼
  */
  private val splitWord = "┼"
  private var wip: String = text

  def replaceAbbreviations: this.type = ???
  def replaceNumbers: this.type = ???
  def replaceContinuousPunctuation: this.type = ???

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