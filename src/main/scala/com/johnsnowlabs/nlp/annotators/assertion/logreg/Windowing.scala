package com.johnsnowlabs.nlp.annotators.assertion.logreg

/**
  * Created by jose on 24/11/17.
  */
trait Windowing {

  val before : Int
  val after : Int

  /* apply window, pad/truncate sentence according to window */
  protected def applyWindow(doc: String, target:String) = {
    val sentSplits = doc.split(target).map(_.trim)
    val targetPart = target.split(" ")

    val leftPart = if (sentSplits.head.isEmpty) Array[String]()
    else sentSplits.head.split(" ")

    val rightPart = if (sentSplits.length == 1) Array[String]()
    else sentSplits.last.split(" ")

    val (start, leftPadding) =
      if(leftPart.size >= before)
        (leftPart.size - before, Array[String]())
      else
        (0, Array.fill(before - leftPart.length)("empty_marker"))

    val (end, rightPadding) =
      if(targetPart.length - 1 + rightPart.length <= after)
        (rightPart.length, Array.fill(after - (targetPart.length - 1 + rightPart.length))("empty_marker"))
      else
        (after - targetPart.length, Array[String]())

    val leftContext = leftPart.slice(start, leftPart.length)
    val rightContext = rightPart.slice(0, end + 1)

    leftPadding ++ leftContext ++ targetPart ++ rightContext ++ rightPadding

  }

}
