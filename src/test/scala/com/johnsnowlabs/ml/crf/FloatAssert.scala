package com.johnsnowlabs.ml.crf

object FloatAssert {

  def seqEquals(a: Seq[Float], b: Seq[Float], eps: Float = 1e-7f): Unit = {
    assert(a.size == b.size, s"$a is not equal $b")

    for (i <- 0 until a.size)
      assert(Math.abs(a(i) - b(i)) <= eps, s"$a does not equal $b\nExpected\t:$b\nActual\t\t:$a\n")
  }

  def equals(a: Float, b: Float, eps: Float = 1e-7f): Unit = {
    assert(Math.abs(a - b) <= eps, s"$a does not equal $b\nExpected\t:$b\nActual\t\t:$a\n")
  }
}
