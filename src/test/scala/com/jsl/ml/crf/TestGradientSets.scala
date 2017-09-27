package com.jsl.ml.crf


object TestGradientSets {

  def small = {
    val metadata = new DatasetMetadata()
    val (label1, word1) = metadata.getFeatures(metadata.startLabel, "label1",
      Seq("one"), Seq("num1" -> 1f, "num2" -> 2f))

    val (label2, word2) = metadata.getFeatures("label1", "label2",
      Seq("two"), Seq("num1" -> 2f, "num2" -> 3f))

    val instance = new Instance(Seq(word1, word2))
    val labels = new InstanceLabels(Seq(1, 2))

    new Dataset(Seq(labels -> instance), metadata)
  }

}
