package com.jsl.ml.crf

import com.jsl.nlp.annotators.param.{SerializedAnnotatorComponent, WritableAnnotatorComponent}


case class AttrStat(val frequency: Int, val sum: Float)


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

  for (i <- 0 until attrs.length)
    require(attrs(i).id == i, s"Attribute ${attrs(i)} stored at index $i that does not equal to id")

  for (i <- 0 until attrFeatures.length)
    require(attrFeatures(i).id == i, s"Feature ${attrFeatures(i)} stored at index $i that does not equal to id")

  // Label Name -> Label Id
  lazy val label2Id = labels.zipWithIndex.toMap

  // Attr Name -> Attr Id
  lazy val attr2Id = attrs.map(attr => (attr.name, attr.id)).toMap

  // (Attr Id, Label Id) -> Feature Id
  lazy val attrFeatures2Id = attrFeatures.map(f => ((f.attrId, f.label), f.id)).toMap

  // Transition -> Feature Id
  lazy val transFeature2Id = {
    transitions
      .zipWithIndex
      .map(p => (p._1, p._2 + attrFeatures.size))
      .toMap
  }

  // Attr Id -> List of AttrFeatures
  lazy val attr2Features = {
    attrFeatures
      .groupBy(f => f.attrId)
      .toSeq
      .sortBy(p => p._1)
      .map(p => p._2)
      .toIndexedSeq
  }

  override def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent] = {
    new SerializedDatasetMetadata(
      labels.toList,
      attrs.toList,
      attrFeatures.toList,
      transitions.toList,
      featuresStat.toList
    )
  }

}

case class SerializedDatasetMetadata
(
  val labels: List[String],
  val attrs: List[Attr],
  val attrFeatures: List[AttrFeature],
  val transitions: List[Transition],
  val featuresStat: List[AttrStat]
)  extends SerializedAnnotatorComponent[DatasetMetadata]
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
