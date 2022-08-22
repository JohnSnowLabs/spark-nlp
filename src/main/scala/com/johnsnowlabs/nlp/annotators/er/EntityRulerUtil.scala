package com.johnsnowlabs.nlp.annotators.er

import scala.collection.mutable.ListBuffer

object EntityRulerUtil {

  def mergeIntervals(intervals: List[List[Int]]): List[List[Int]] = {

    val mergedIntervals = ListBuffer[List[Int]]()
    var currentMergedInterval = List[Int]()
    val sortedIntervals = intervals.sortBy(interval => interval.head)

    sortedIntervals.zipWithIndex.foreach { case (interval, index) =>
      if (index == 0) {
        currentMergedInterval = interval
      } else {
        val mergedEnd = currentMergedInterval(1)
        val currentBegin = interval.head
        if (mergedEnd >= currentBegin) {
          val currentEnd = interval(1)
          val maxEnd = math.max(currentEnd, mergedEnd)
          currentMergedInterval = List(currentMergedInterval.head, maxEnd)
        } else {
          mergedIntervals.append(currentMergedInterval)
          currentMergedInterval = interval
        }
      }
    }

    mergedIntervals.append(currentMergedInterval)
    mergedIntervals.toList

  }

}
