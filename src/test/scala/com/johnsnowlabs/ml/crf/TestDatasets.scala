/*
 * Copyright 2017-2019 John Snow Labs
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

object TestDatasets {

  def smallText = {
    val labels = new TextSentenceLabels(Seq("One", "Two", "One", "Two"))

    val sentence = new TextSentenceAttrs(Seq(
      new WordAttrs(Seq("attr1" -> "")),
      new WordAttrs(Seq("attr1" -> "value1", "attr2" ->"value1", "attr3" -> "")),
      new WordAttrs(Seq("attr1" -> "", "attr3" -> "")),
      new WordAttrs(Seq("attr1" -> "value1"))
    ))
    Seq(labels -> sentence).toIterator
  }

  def doubleText = smallText ++ smallText

  def small = {
    val metadata = new DatasetEncoder()
    val (label1, word1) = metadata.getFeatures(metadata.startLabel, "label1",
      Seq("one"), Seq(1f, 2f))

    val (label2, word2) = metadata.getFeatures("label1", "label2",
      Seq("two"), Seq(2f, 3f))

    val instance = new Instance(Seq(word1, word2))
    val labels = new InstanceLabels(Seq(1, 2))

    new CrfDataset(Seq(labels -> instance), metadata.getMetadata)
  }

}