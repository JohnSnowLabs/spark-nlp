package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.{Map => MMap}

class TupleKeyLongDoubleMapAccumulator(defaultMap: MMap[(String, String), (Long, Double)] = MMap.empty[(String, String), (Long, Double)])
  extends AccumulatorV2[((String, String), (Long, Double)), Map[(String, String), (Long, Double)]] {

  var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: ((String, String), (Long, Double))): Unit = mmap(v._1) = v._2

  def updateMany(other: MMap[(String, String), (Long, Double)]): Unit = {
    other.foreach{case (k, v) =>
      mmap(k) = mmap.get(k).map{case (v1, v2) => (Seq(v1, v._1).max, v2 + v._2)}.getOrElse(v)
    }
  }

  def update(k: (String, String), v: (Long, Double)): Unit =  mmap(k) = v

  override def value: Map[(String, String), (Long, Double)] = mmap.toMap

  override def copy(): AccumulatorV2[((String, String), (Long, Double)), Map[(String, String), (Long, Double)]] = {
    val c = new TupleKeyLongDoubleMapAccumulator(MMap.empty[(String, String), (Long, Double)])
    c.mmap = this.mmap
    c
  }


  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[((String, String), (Long, Double)), Map[(String, String), (Long, Double)]]): Unit = {
    other match {
      case o: TupleKeyLongDoubleMapAccumulator =>
        updateMany(o.mmap)
      case _ => throw new Exception("Cannot merge tuple key long")
    }
  }
}

class StringMapStringDoubleAccumulator(defaultMap: MMap[String, MMap[String, Double]] = MMap.empty[String, MMap[String, Double]])
  extends AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]] {

  private var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, MMap[String, Double])): Unit = {
    mmap.update(v._1, mmap(v._1) ++ v._2)
  }

  override def value: Map[String, Map[String, Double]] = mmap.mapValues(_.toMap).toMap

  override def copy(): AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]] = {
    val c = new StringMapStringDoubleAccumulator(MMap.empty[String, MMap[String, Double]])
    c.mmap ++= this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  def addMany(other: MMap[String, MMap[String, Double]]) = {
    other.foreach { case (k, v) => v.foreach { case (kk, vv) =>
      mmap(k) = mmap.getOrElse(k, MMap()) ++ MMap(kk -> vv)
    }}
  }

  override def merge(other: AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]]): Unit =
    other match {
      case o: StringMapStringDoubleAccumulator =>
        addMany(o.mmap)
      case _ => throw new Exception("Wrong StringMapStringDouble merge")
    }
}