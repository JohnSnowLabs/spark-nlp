package com.johnsnowlabs.ml.crf

import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

import java.io._


class SerializationSpec extends FlatSpec {
  val dataset = TestDatasets.small
  val metadata = dataset.metadata
  val weights = (0 until 8).map(i => 1f / i).toArray

  val model = new LinearChainCrfModel(weights, dataset.metadata)

  "LinearChainCrfModel" should "serialize and deserialize correctly" taggedAs FastTest in {
    val memory = new ByteArrayOutputStream()

    val oos = new ObjectOutputStream(memory)
    oos.writeObject(model)
    oos.close

    val input = new ObjectInputStream(new ByteArrayInputStream(memory.toByteArray))
    val deserialized = input.readObject().asInstanceOf[LinearChainCrfModel]

    assert(deserialized.weights.toSeq == model.weights.toSeq)

    val newMeta = deserialized.metadata
    assert(newMeta.labels.toSeq == metadata.labels.toSeq)
    assert(newMeta.attrs.toSeq == metadata.attrs.toSeq)
    assert(newMeta.attrFeatures.toSeq == metadata.attrFeatures.toSeq)
    assert(newMeta.transitions.toSeq == metadata.transitions.toSeq)
    assert(newMeta.featuresStat.toSeq == metadata.featuresStat.toSeq)
  }
}
