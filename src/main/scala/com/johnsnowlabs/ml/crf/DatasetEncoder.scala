package com.johnsnowlabs.ml.crf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class DatasetEncoder(val startLabel: String = "@#Start") {

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

  // transition -> freq
  val transFeaturesFreq = mutable.Map[Transition, Int]()

  // All transitions
  def transitions = transFeaturesFreq.keys.toSeq


  private def addAttrFeature(label: Int, attr: Int, value: Float): Unit = {
    val featureId = attrFeatures2Id.getOrElseUpdate((attr, label), attrFeatures2Id.size)

    if (featureId >= attrFeatures.size) {
      val feature = AttrFeature(featureId, attr, label)
      attrFeatures.append(feature)
      attrFeaturesFreq.append(0)
      attrFeaturesSum.append(0f)
    }

    attrFeaturesFreq(featureId) += 1
    attrFeaturesSum(featureId) += value
  }

  private def addTransFeature(fromId: Int, toId: Int): Unit = {
    val meta = Transition(fromId, toId)
    transFeaturesFreq(meta) = transFeaturesFreq.getOrElse(meta, 0) + 1
  }

  private def getLabel(label: String): Int = {
    label2Id.getOrElseUpdate(label, label2Id.size)
  }

  private def getAttr(attr: String, isNumerical: Boolean): Int = {
    val attrId = attr2Id.getOrElseUpdate(attr, attr2Id.size)
    if (attrId >= attributes.size) {
      attributes.append(
        Attr(attrId, attr, isNumerical))
    }

    attrId
  }

  def getFeatures(prevLabel: String = startLabel,
                  label: String,
                  binaryAttrs: Seq[String],
                  numAttrs: Seq[Float]): (Int, SparseArray) = {
    val labelId = getLabel(label)

    val binFeature = binaryAttrs.map{attr =>
      val attrId = getAttr(attr, false)
      addAttrFeature(labelId, attrId, 1f)
      (attrId, 1f)
    }

    val numFeatures = numAttrs.zipWithIndex.map{case(value, idx) => {
      val attrId = getAttr("num" + idx, true)
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

  def getMetadata: DatasetMetadata = {
    val labels = label2Id.toSeq.sortBy(a => a._2).map(a => a._1).toArray
    val transitionsStat = transitions
      .map(transition => transFeaturesFreq(transition))
      .map(freq => new AttrStat(freq, freq))

    val attrsStat = attrFeaturesFreq
      .zip(attrFeaturesSum)
      .map(p => new AttrStat(p._1, p._2))

    new DatasetMetadata(
      labels,
      copy(attributes),
      copy(attrFeatures),
      transitions.toArray,
      (attrsStat ++ transitionsStat).toArray
    )
  }

  private def copy[T : ClassTag](source: IndexedSeq[T]): Array[T] = {
    if (source.length == 0) {
      Array.empty[T]
    } else {
      val first = source(0)
      val result = Array.fill(source.length)(first)
      for (i <- 0 until source.length)
        result(i) = source(i)

      result
    }
  }
}
