/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
