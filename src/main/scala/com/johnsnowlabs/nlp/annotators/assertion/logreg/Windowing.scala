package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.embeddings.{AnnotatorWithWordEmbeddings, WordEmbeddings}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Created by jose on 24/11/17.
  */
trait Windowing {

  val before : Int = 6
  val after : Int = 18
  val embeddingsPath: String = "override me"

  // hard-coded stuff
  lazy val wordVectors: Option[WordEmbeddings] = None //= Some(WordEmbeddings(embeddingsPath, 200))


  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, target:String) : Array[String] = {
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

  /* same as above, but convert the resulting text in a vector */
  def applyWindowUdf =
    udf {(doc:String, target:String)  =>
      val tmp : Array[Double] = applyWindow(doc.toLowerCase, target.toLowerCase).flatMap(wordVectors.get.getEmbeddings).map(_.toDouble)

      /* TODO tmp sanity check - remove */
      if (tmp.length != (before + after + 1) * 200)
        println(doc)

      if (tmp.contains(Double.NaN))
        println(doc)

      Vectors.dense(tmp)

    }

  /* Column label must be of type NumericType but was actually of type StringType. */
  def labelToNumber() = udf { label:String  =>
    if (label.equals("Affirmed")) 1.0 else 0.0
  }
}
