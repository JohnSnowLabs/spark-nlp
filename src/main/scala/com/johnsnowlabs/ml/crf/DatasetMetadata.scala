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

import com.johnsnowlabs.nlp.annotators.param.{SerializedAnnotatorComponent, WritableAnnotatorComponent}

case class AttrStat(frequency: Int, sum: Float)


class DatasetMetadata
(
  // labelId -> Label Name
  val labels: Array[String],

  // AttrId -> Attr Metadata
  val attrs: Array[Attr],

  // FeatureId -> AttrFeature
  val attrFeatures: Array[AttrFeature],

  // All possible transitions according to train dataset
  val transitions: Array[Transition],

  // FeatureId -> (attr, label) statistics
  val featuresStat: Array[AttrStat]

) extends WritableAnnotatorComponent {

  require(attrFeatures.length + transitions.length == featuresStat.length,
    s"Number of features ${featuresStat.length} should be equal to number of attr features ${attrFeatures.length}" +
      s" plus number of transition features ${transitions.length}")

  for (i <- attrs.indices)
    require(attrs(i).id == i, s"Attribute ${attrs(i)} stored at index $i that does not equal to id")

  for (i <- attrFeatures.indices)
    require(attrFeatures(i).id == i, s"Feature ${attrFeatures(i)} stored at index $i that does not equal to id")

  // Label Name -> Label Id
  lazy val label2Id: Map[String, Int] = labels.zipWithIndex.toMap

  // Attr Name -> Attr Id
  lazy val attr2Id: Map[String, Int] = attrs.map(attr => (attr.name, attr.id)).toMap

  // (Attr Id, Label Id) -> Feature Id
  lazy val attrFeatures2Id: Map[(Int, Int), Int] = attrFeatures.map(f => ((f.attrId, f.label), f.id)).toMap

  // Transition -> Feature Id
  lazy val transFeature2Id: Map[Transition, Int] = {
    transitions
      .zipWithIndex
      .map(p => (p._1, p._2 + attrFeatures.length))
      .toMap
  }

  // Attr Id -> List of AttrFeatures
  lazy val attr2Features: IndexedSeq[Array[AttrFeature]] = {
    val attr2FeaturesMap = attrFeatures
      .groupBy(f => f.attrId)
      .toSeq
      .sortBy(p => p._1)
      .toMap

    (0 until attrs.length).map{ aId =>
      attr2FeaturesMap.getOrElse(aId, Array.empty)
    }
  }

  override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = {
    SerializedDatasetMetadata(
      labels.toList,
      attrs.toList,
      attrFeatures.toList,
      transitions.toList,
      featuresStat.toList
    )
  }

  /**
    * Leaves only features that in featureIds list
    * @param featureIds - feature ids to leave
   */
  def filterFeatures(featureIds: Seq[Int]): DatasetMetadata = {
    val (attrFeaturesIds, transFeaturesIds) = featureIds.partition(id => id < attrFeatures.length)
    val filteredAttrFeatures = attrFeaturesIds
      .map(id => attrFeatures(id))
      .zipWithIndex
      .map{case(oldAttr, idx) => AttrFeature(idx, oldAttr.attrId, oldAttr.label)}
      .toArray

    val filteredTransFeatures = transFeaturesIds.map(id => transitions(id - attrFeatures.length)).toArray
    val filteredStat = featureIds.map(id => featuresStat(id)).toArray

    new DatasetMetadata(labels, attrs, filteredAttrFeatures, filteredTransFeatures, filteredStat)
  }
}

case class SerializedDatasetMetadata
(
  labels: Seq[String],
  attrs: Seq[Attr],
  attrFeatures: Seq[AttrFeature],
  transitions: Seq[Transition],
  featuresStat: Seq[AttrStat]
)
  extends SerializedAnnotatorComponent[DatasetMetadata]
{
  override def deserialize: DatasetMetadata = {
    new DatasetMetadata(
      labels.toArray,
      attrs.toArray,
      attrFeatures.toArray,
      transitions.toArray,
      featuresStat.toArray
    )
  }
}
