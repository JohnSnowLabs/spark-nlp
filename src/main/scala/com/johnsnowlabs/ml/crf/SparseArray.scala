package com.johnsnowlabs.ml.crf

class SparseArray(val values: Array[(Int, Float)]) {
  var prev = -1
  for ((idx, _) <- values) {
    require(idx > prev, s"index must be sorted $prev must be lower $idx")
    prev = idx
  }

  def apply(idx: Int): Float = binSearch(idx)

  private def binSearch(idx: Int, l: Int = 0, r: Int = values.length - 1): Float = {
    val mid = (l + r) / 2

    if (l > r)
      0f
    else if (idx == values(mid)._1)
      values(mid)._2
    else if (idx < values(mid)._1)
      binSearch(idx, l, mid - 1)
    else
      binSearch(idx, mid + 1, r)
  }

}

object SparseArray {
  implicit class SeqWrapper(values: Seq[(Int, Float)]) {
    def toSparse(): SparseArray = {
      new SparseArray(values.sortBy(_._1).toArray)
    }
  }
}