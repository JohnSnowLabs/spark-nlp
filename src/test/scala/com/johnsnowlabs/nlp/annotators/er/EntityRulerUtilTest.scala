package com.johnsnowlabs.nlp.annotators.er

import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerUtilTest extends AnyFlatSpec {

  "EntityRulerUtil" should "merge intervals" in {

    var intervals =
      List(List(57, 59), List(7, 10), List(7, 15), List(57, 64), List(12, 15), List(0, 15))
    var expectedMerged = List(List(0, 15), List(57, 64))

    var actualMerged = EntityRulerUtil.mergeIntervals(intervals)

    assert(expectedMerged == actualMerged)

    intervals = List(List(2, 3), List(4, 5), List(6, 7), List(8, 9), List(1, 10))
    expectedMerged = List(List(1, 10))

    actualMerged = EntityRulerUtil.mergeIntervals(intervals)
    assert(expectedMerged == actualMerged)

    intervals = List(List(5, 10), List(15, 20))
    expectedMerged = List(List(5, 10), List(15, 20))

    actualMerged = EntityRulerUtil.mergeIntervals(intervals)
    assert(expectedMerged == actualMerged)
  }

}
