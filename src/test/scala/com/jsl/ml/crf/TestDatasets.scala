package com.jsl.ml.crf

object TestDatasets {

  def small = {
    val labels = new TextSentenceLabels(Seq("One", "Two", "One", "Two"))

    val sentence = new TextSentence(Seq(
      new TextToken(Seq("attr1" -> "")),
      new TextToken(Seq("attr1" -> "value1", "attr2" ->"value1", "attr3" -> "")),
      new TextToken(Seq("attr1" -> "", "attr3" -> "")),
      new TextToken(Seq("attr1" -> "value1"))
    ))
    Seq(labels -> sentence).toIterator
  }

  def double = small ++ small
}