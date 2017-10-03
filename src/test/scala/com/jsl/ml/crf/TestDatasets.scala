package com.jsl.ml.crf

object TestDatasets {

  def smallText = {
    val labels = new TextSentenceLabels(Seq("One", "Two", "One", "Two"))

    val sentence = new TextSentenceAttrs(Seq(
      new TextAttr(Seq("attr1" -> "")),
      new TextAttr(Seq("attr1" -> "value1", "attr2" ->"value1", "attr3" -> "")),
      new TextAttr(Seq("attr1" -> "", "attr3" -> "")),
      new TextAttr(Seq("attr1" -> "value1"))
    ))
    Seq(labels -> sentence).toIterator
  }

  def doubleText = smallText ++ smallText

  def small = {
    val metadata = new DatasetEncoder()
    val (label1, word1) = metadata.getFeatures(metadata.startLabel, "label1",
      Seq("one"), Seq("num1" -> 1f, "num2" -> 2f))

    val (label2, word2) = metadata.getFeatures("label1", "label2",
      Seq("two"), Seq("num1" -> 2f, "num2" -> 3f))

    val instance = new Instance(Seq(word1, word2))
    val labels = new InstanceLabels(Seq(1, 2))

    new CrfDataset(Seq(labels -> instance), metadata.getMetadata)
  }

}