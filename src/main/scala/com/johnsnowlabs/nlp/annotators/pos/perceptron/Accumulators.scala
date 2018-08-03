package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.{ArrayBuffer, Map => MMap}

class TupleKeyLongDoubleMapAccumulator(defaultMap: MMap[(String, String), (Long, Double)] = MMap.empty[(String, String), (Long, Double)])
  extends AccumulatorV2[((String, String), (Long, Double)), Map[(String, String), (Long, Double)]] {

  val mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: ((String, String), (Long, Double))): Unit = {
    mmap(v._1) = mmap.get(v._1).map{case (v1, v2) => ((v1 + v._2._1)/2, (v2 + v._2._2)/2)}.getOrElse(v._2)
  }

  def updateMany(other: MMap[(String, String), (Long, Double)]): Unit = {
    other.foreach{case (k, v) =>
      this.add((k, v))
    }
  }

  override def value: Map[(String, String), (Long, Double)] = mmap.toMap

  override def copy(): AccumulatorV2[((String, String), (Long, Double)), Map[(String, String), (Long, Double)]] = {
    val m = ArrayBuffer.empty[((String, String), (Long, Double))]
    this.mmap.copyToBuffer(m)
    new TupleKeyLongDoubleMapAccumulator(MMap(m:_*))
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

  private val mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, MMap[String, Double])): Unit = {
    v._2.foreach{case (kk, vv) =>
      val loc = mmap.getOrElse(v._1, MMap.empty[String, Double])
      val nv = if (loc.isDefinedAt(kk)) (loc.getOrElse(kk, 0.0) + vv) / 2.0 else vv
      mmap.update(v._1, loc.updated(kk, nv))
    }
  }

  override def value: Map[String, Map[String, Double]] = mmap.mapValues(_.toMap.filterNot(a => a._2 == 0)).toMap

  override def copy(): AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]] = {
    val m = ArrayBuffer.empty[(String, MMap[String, Double])]
    this.mmap.copyToBuffer(m)
    new StringMapStringDoubleAccumulator(MMap(m:_*))
  }

  override def isZero: Boolean = mmap.isEmpty

  def addMany(other: MMap[String, MMap[String, Double]]) = {
    other.foreach { case (k,v) =>
      this.add((k,v))
    }
  }

  override def merge(other: AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]]): Unit = {
    other match {
      case o: StringMapStringDoubleAccumulator =>
        addMany(o.mmap)
      case _ => throw new Exception("Wrong StringMapStringDouble merge")
    }
  }
}