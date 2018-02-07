package com.johnsnowlabs.util.spark

import org.apache.spark.util.AccumulatorV2
import scala.collection.mutable.{Map=>MMap}

class MapAccumulator(defaultMap: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0))
  extends AccumulatorV2[(String, Long), Map[String, Long]] {

  private val mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, Long)): Unit = mmap(v._1) += v._2

  override def value: Map[String, Long] = mmap.toMap

  override def copy(): AccumulatorV2[(String, Long), Map[String, Long]] = new MapAccumulator(MMap[String, Long](value.toSeq:_*))

  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[(String, Long), Map[String, Long]]): Unit = other.value.foreach{case (k, v) => mmap(k) += v}

}
