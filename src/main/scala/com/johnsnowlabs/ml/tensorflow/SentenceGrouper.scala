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

package com.johnsnowlabs.ml.tensorflow

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

case class SentenceGrouper[T: ClassTag]
(
  getLength: T => Int,
  sizes: Array[Int] = Array(5, 10, 20, 50)
) {

  def getBucketId(len: Int): Int = {
    for (i <- 0 until sizes.length) {
      if (len <= sizes(i))
        return i
    }

    sizes.length
  }

  def slice(source: TraversableOnce[T], batchSize: Int = 32): Iterator[Array[T]] = {
    val buckets = Array.fill(sizes.length + 1)(ArrayBuffer.empty[T])

    val batches = source.toIterator.flatMap{item =>
      val length = getLength(item)
      val bucketId = getBucketId(length)
      buckets(bucketId).append(item)
      if (buckets(bucketId).length >= batchSize) {
        val result = buckets(bucketId).toArray
        buckets(bucketId).clear()

        Some(result)
      }
      else {
        None
      }
    }

    val rest = buckets.toIterator.filter(b => b.nonEmpty).map(b => b.toArray)

    batches ++ rest
  }
}
