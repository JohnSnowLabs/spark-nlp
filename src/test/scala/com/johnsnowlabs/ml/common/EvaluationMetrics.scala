package com.johnsnowlabs.ml.common


/**
  * Created by jose on 19/12/17.
  */
trait EvaluationMetrics {

  case class TpFnFp(tp: Int, fn: Int, fp: Int)

  def confusionMatrix[T](predicted: Seq[T], gold: Seq[T]) = {
    val labels = gold.distinct
    import scala.collection.mutable.{Map => MutableMap}
    val matrix : Map[T, MutableMap[T, Int]] =
      labels.map(label => (label, MutableMap(labels.zip(Array.fill(labels.size)(0)): _*))).toMap

    predicted.zip(gold).foreach { case (p, g) => matrix.get(p).get(g) += 1}

    /* sanity check, the confusion matrix should contain as many elements as there were used during training / prediction */
    assert(predicted.length ==matrix.map(map => map._2.values.sum).sum)
    matrix
  }


  def calcStat[T](predicted: Seq[T], gold: Seq[T]):(Float, Float, Float) = {
    val tpFnFp = predicted.zip(gold).map({case (p, g) =>
      if (p == g)
        TpFnFp(1, 0, 0)
      else
        TpFnFp(0, 1, 1)
    }).reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

    calcStat(tpFnFp.tp + tpFnFp.fn, tpFnFp.tp + tpFnFp.fp, tpFnFp.tp)
  }

  def calcStat(correct: Long, predicted: Long, predictedCorrect: Long): (Float, Float, Float) = {
    val prec = predictedCorrect.toFloat / predicted
    val rec = predictedCorrect.toFloat / correct
    val f1 = 2 * prec * rec / (prec + rec)
    (prec, rec, f1)
  }


}
