package com.johnsnowlabs.util.spark

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.{Map => MMap}

class LongMapAccumulatorWithDefault(defaultMap: MMap[String, Long] = MMap.empty[String, Long], defaultValue: Long = 0L)
  extends AccumulatorV2[(String, Long), Map[String, Long]] {

  private var mmap = defaultMap.withDefaultValue(defaultValue)

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, Long)): Unit = mmap(v._1) += v._2

  override def value: Map[String, Long] = mmap.toMap.withDefaultValue(defaultValue)

  override def copy(): AccumulatorV2[(String, Long), Map[String, Long]] = {
    val c = new LongMapAccumulatorWithDefault(MMap[String, Long](value.toSeq: _*), defaultValue)
    c.mmap = this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[(String, Long), Map[String, Long]]): Unit =
    mmap = mmap ++ other.value
}

class DoubleMapAccumulatorWithDefault(defaultMap: MMap[String, Double] = MMap.empty[String, Double], defaultValue: Double = 0.0)
  extends AccumulatorV2[(String, Double), Map[String, Double]] {

  @volatile private var mmap = defaultMap.withDefaultValue(defaultValue)

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, Double)): Unit = mmap(v._1) = mmap(v._1) + v._2

  override def value: Map[String, Double] = mmap.toMap.withDefaultValue(defaultValue)

  override def copy(): AccumulatorV2[(String, Double), Map[String, Double]] = {
    val c = new DoubleMapAccumulatorWithDefault(MMap.empty[String, Double], defaultValue)
    c.mmap = this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[(String, Double), Map[String, Double]]): Unit =
    mmap ++= other.value
}

class TupleKeyLongMapAccumulatorWithDefault(defaultMap: MMap[(String, String), Long] = MMap.empty[(String, String), Long], defaultValue: Long = 0L)
  extends AccumulatorV2[((String, String), Long), Map[(String, String), Long]] {

  var mmap = defaultMap.withDefaultValue(defaultValue)

  override def reset(): Unit = mmap.clear()

  override def add(v: ((String, String), Long)): Unit = mmap(v._1) = v._2

  def updateMany(v: MMap[(String, String), Long]): Unit = {
    mmap ++= v
  }

  def update(k: (String, String), v: Long): Unit =  mmap(k) = v

  override def value: Map[(String, String), Long] = mmap.toMap.withDefaultValue(defaultValue)

  override def copy(): AccumulatorV2[((String, String), Long), Map[(String, String), Long]] = {
    val c = new TupleKeyLongMapAccumulatorWithDefault(MMap.empty[(String, String), Long], defaultValue)
    c.mmap = this.mmap
    c
  }


  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[((String, String), Long), Map[(String, String), Long]]): Unit =
    other match {
      case o: TupleKeyLongMapAccumulatorWithDefault =>
        mmap = mmap ++ o.mmap
      case _ => throw new Exception("Cannot merge tuple key long")
    }
}