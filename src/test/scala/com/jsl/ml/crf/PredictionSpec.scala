package com.jsl.ml.crf

import org.scalatest.FlatSpec

class PredictionSpec extends FlatSpec {

  "CrfModel" should "return correct prediction" in {
    val dataset = TestDatasets.small
    val weights = Array.fill(8)(0.1f)

    val model = new LinearChainCrfModel(weights, dataset.metadata)
    val instance = dataset.instances.head._2
    val labels = model.predict(instance)

    assert(labels.labels == Seq(1, 2))
  }

  "CrfModel" should "return correct prediction with negative sums" in {
    val dataset = TestDatasets.small
    val weights = Array.fill(8)(-0.1f)

    val model = new LinearChainCrfModel(weights, dataset.metadata)
    val instance = dataset.instances.head._2
    val labels = model.predict(instance)

    assert(labels.labels == Seq(0, 0))
  }
}

