package com.jsl.ml.crf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

// ToDo redesign splitting Mutable and Immutable
class DatasetMetadata(val startLabel: String = "@#Start") {

  // Attr Name -> AttrId
  val attr2Id = mutable.Map[String, Int]()

  // Label Name -> labelId
  val label2Id = mutable.Map[String, Int](startLabel -> 0)

  // AttrId -> Attr
  val attributes = mutable.ArrayBuffer[Attr]()

  // (attrId, labelId) -> featureId
  val attrFeatures2Id = mutable.Map[(Int, Int), Int]()

  // featureId -> AttrFeature
  val attrFeatures = ArrayBuffer[AttrFeature]()

  // featureId -> freq
  val attrFeaturesFreq = ArrayBuffer[Int]()

  // featureId -> Sum
  val attrFeaturesSum = ArrayBuffer[Float]()

  // Map attrId -> List of AttrFeatures
  val attr2Features = ArrayBuffer[ArrayBuffer[AttrFeature]]()


  // transition -> freq
  val transFeaturesFreq = mutable.Map[Transition, Int]()

  // All transitions
  def transitions = transFeaturesFreq.keys

  // ToDo Redisign
  // transition -> featureId
  lazy val transFeature2Id = {
    transitions
      .zipWithIndex
      .map(p => (p._1, p._2 + attrFeatures.size))
      .toMap
  }

  private def addAttrFeature(label: Int, attr: Int, value: Float): Unit = {
    val featureId = attrFeatures2Id.getOrElseUpdate((attr, label), attrFeatures2Id.size)

    if (featureId >= attrFeatures.size) {
      val feature = new AttrFeature(featureId, attr, label)
      attrFeatures.append(feature)
      attrFeaturesFreq.append(0)
      attrFeaturesSum.append(0f)
      attr2Features(attr).append(feature)
    }

    attrFeaturesFreq(featureId) += 1
    attrFeaturesSum(featureId) += value
  }

  private def addTransFeature(fromId: Int, toId: Int): Unit = {
    val meta = new Transition(fromId, toId)
    transFeaturesFreq(meta) = transFeaturesFreq.getOrElse(meta, 0) + 1
  }

  private def getLabel(label: String): Int = {
    label2Id.getOrElseUpdate(label, label2Id.size)
  }

  private def getAttr(attr: String, isNumerical: Boolean): Int = {
    val attrId = attr2Id.getOrElseUpdate(attr, attr2Id.size)
    if (attrId >= attributes.size) {
      attributes.append(new Attr(attrId, attr, isNumerical))
      attr2Features.append(ArrayBuffer.empty)
    }

    attrId
  }

  def getFeatures(prevLabel: String = startLabel,
                  label: String,
                  binaryAttrs: Seq[String],
                  numAttrs: Seq[(String, Float)]): (Int, SparseArray) = {
    val labelId = getLabel(label)

    val binFeature = binaryAttrs.map{attr =>
      val attrId = getAttr(attr, false)
      addAttrFeature(labelId, attrId, 1f)
      (attrId, 1f)
    }

    val numFeatures = numAttrs.map{case(attr, value) => {
      val attrId = getAttr(attr, true)
      addAttrFeature(labelId, attrId, value)
      (attrId, value)
    }}

    val fromId = getLabel(prevLabel)
    addTransFeature(fromId, labelId)

    val features = (binFeature ++ numFeatures)
      .sortBy(_._1)
      .distinct
      .toArray

    (labelId, new SparseArray(features))
  }
}
