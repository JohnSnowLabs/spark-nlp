/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.util.spark

import org.apache.spark.util.AccumulatorV2
import scala.collection.mutable.{Map=>MMap}

class MapAccumulator(defaultMap: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0))
  extends AccumulatorV2[(String, Long), Map[String, Long]] {

  private val _mmap = defaultMap

  override def reset(): Unit = _mmap.clear()

  override def add(v: (String, Long)): Unit = _mmap(v._1) += v._2

  override def value: Map[String, Long] = _mmap.toMap.withDefaultValue(0)

  override def copy(): AccumulatorV2[(String, Long), Map[String, Long]] =
    new MapAccumulator(MMap[String, Long](value.toSeq:_*).withDefaultValue(0))

  override def isZero: Boolean = _mmap.isEmpty

  def mmap = _mmap

  override def merge(other: AccumulatorV2[(String, Long), Map[String, Long]]): Unit =
    other match {
      case o: MapAccumulator => o.mmap.foreach{case (k, v) => _mmap(k) += v}
      case _ => throw new UnsupportedOperationException(
        s"Cannot merge ${this.getClass.getName} with ${other.getClass.getName}")
    }

}
