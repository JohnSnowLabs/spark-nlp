package com.johnsnowlabs.util.spark

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.{Map => MMap}

class LongMapAccumulator(defaultMap: MMap[String, Long] = MMap.empty[String, Long])
  extends AccumulatorV2[(String, Long), Map[String, Long]] {

  private var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, Long)): Unit = mmap(v._1) += v._2

  override def value: Map[String, Long] = mmap.toMap

  override def copy(): AccumulatorV2[(String, Long), Map[String, Long]] = {
    val c = new LongMapAccumulator(MMap[String, Long](value.toSeq: _*))
    c.mmap = this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[(String, Long), Map[String, Long]]): Unit =
    mmap = mmap ++ other.value
}