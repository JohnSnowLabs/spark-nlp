package com.jsl.ml.crf

import org.scalatest.FlatSpec


class DatasetSpec extends FlatSpec {

  "DatasetReader" should "correct calculate transit freaquences" in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.small)
    val metadata = dataset.metadata

    assert(metadata.label2Id.size == 3)
    assert(metadata.transitions.size == 3)

    val features = metadata.transitions
    assert(features.toSet == Seq(Transition(0, 1), Transition(1, 2), Transition(2, 1)).toSet)

    val featureFreqs = metadata.transFeaturesFreq

    assert(featureFreqs(Transition(0, 1)) ==  1)
    assert(featureFreqs(Transition(1, 2)) ==  2)
    assert(featureFreqs(Transition(2, 1)) ==  1)
  }


  "DatasetReader" should "correct read fill attrs metadata" in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.double)
    val metadata = dataset.metadata

    assert(metadata.attr2Id.size == 4)

    val attrFeatures = metadata.attributes
    assert(attrFeatures.size == 4)

    assert(attrFeatures(0) == new Attr(0, "attr1=", false))
    assert(metadata.attr2Features(0) == Seq(AttrFeature(0, 0, 1)))
    assert(metadata.attr2Features(1) == Seq(AttrFeature(1, 1, 2)))
    assert(metadata.attr2Features(3).toSet == Seq(AttrFeature(3, 3, 2), AttrFeature(4, 3, 1)).toSet)
  }


  "DatasetReader" should "correct fill Dataset features" in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.double)
    val instances = dataset.instances

    assert(instances.size == 2)

    val firstLabels = instances(0)._1.labels
    assert(firstLabels == Seq(1, 2, 1, 2))

    val firstWords = instances(0)._2.items

    assert(firstWords.size == 4)

    // Check sparse feature vectors for first instance
    assert(firstWords(0).values.toSeq == Seq(0 -> 1f))
    assert(firstWords(1).values.toSeq == Seq(1 -> 1f, 2 -> 1f, 3 -> 1f))
    assert(firstWords(2).values.toSeq == Seq(0 -> 1f, 3 -> 1f))
    assert(firstWords(3).values.toSeq == Seq(1 -> 1f))
  }
}
