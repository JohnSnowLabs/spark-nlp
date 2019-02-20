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
