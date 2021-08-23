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


class DatasetSpec extends FlatSpec {

  "DatasetReader" should "correct calculate transit freaquences" taggedAs FastTest in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.smallText)
    val metadata = dataset.metadata

    assert(metadata.label2Id.size == 3)
    assert(metadata.transitions.size == 3)

    val transitions = Seq(Transition(0, 1), Transition(1, 2), Transition(2, 1))
    assert(metadata.transitions.toSet == transitions.toSet)

    val frequencies = transitions.map(transition => {
      val id = metadata.transFeature2Id(transition)
      metadata.featuresStat(id).frequency
    })

    assert(frequencies == Seq(1, 2, 1))
  }


  "DatasetReader" should "correct read fill attrs metadata" taggedAs FastTest in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.doubleText)
    val metadata = dataset.metadata

    assert(metadata.attr2Id.size == 4)

    assert(metadata.attrs.size == 4)

    assert(metadata.attrs(0) == new Attr(0, "attr1=", false))
    assert(metadata.attr2Features(0).toSet == Set(AttrFeature(0, 0, 1)))
    assert(metadata.attr2Features(1).toSet == Set(AttrFeature(1, 1, 2)))
    assert(metadata.attr2Features(3).toSet == Set(AttrFeature(3, 3, 2), AttrFeature(4, 3, 1)))
  }


  "DatasetReader" should "correct fill Dataset features" taggedAs FastTest in {
    val dataset = DatasetReader.encodeDataset(TestDatasets.doubleText)
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
