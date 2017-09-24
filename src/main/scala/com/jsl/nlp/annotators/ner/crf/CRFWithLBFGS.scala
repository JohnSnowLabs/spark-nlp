package com.jsl.nlp.annotators.ner.crf

import scala.collection.mutable
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN => BreezeOWLQN, LBFGS => BreezeLBFGS}
import breeze.linalg.{DenseVector => BDV, sum => Bsum}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import scala.language.existentials

class CRFWithLBFGS(private var gradient: CRFGradient, private var updater: Updater)
  extends LBFGS(gradient: Gradient, updater: Updater) {

  private val numCorrections = 5
  private var maxNumIterations = 100
  private var convergenceTol = 1E-4
  private var regParam = 0.5

  /**
   * Set the regularization parameter. Default 0.5.
   */
  override def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance of iterations for L-BFGS. Default 1E-4.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * This value must be nonnegative. Lower convergence values are less tolerant
   * and therefore generally cause more iterations to be run.
   */
  override def setConvergenceTol(tolerance: Double): this.type = {
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the maximal number of iterations for L-BFGS. Default 100.
   */
  override def setNumIterations(iters: Int): this.type = {
    this.maxNumIterations = iters
    this
  }

  def optimizer(data: RDD[Tagger], initialWeights: BDV[Double]): BDV[Double] = {
    CRFWithLBFGS.runLBFGS(
      data,
      gradient,
      updater,
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeights
    )
  }
}

object CRFWithLBFGS {
  def runLBFGS(
    data: RDD[Tagger],
    gradient: CRFGradient,
    updater: Updater,
    numCorrections: Int,
    convergenceTol: Double,
    maxNumIterations: Int,
    regParam: Double,
    initialWeights: BDV[Double]
  ): BDV[Double] = {

    val costFun = new CostFun(data, gradient, updater, regParam)

    var lbfgs: BreezeLBFGS[BDV[Double]] = null

    updater match {
      case updater: L1Updater =>
        lbfgs = new BreezeOWLQN[Int, BDV[Double]](maxNumIterations, numCorrections, regParam, convergenceTol)
      case updater: L2Updater =>
        lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)
    }

    val states = lbfgs.iterations(new CachedDiffFunction[BDV[Double]](costFun), initialWeights)

    val lossHistory = mutable.ArrayBuilder.make[Double]
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }

    println("LBFGS.runLBFGS finished after %s iterations. last 10 losses: %s".format(
      state.iter, lossHistory.result().takeRight(10).mkString(" -> ")
    ))
    state.x
  }
}

class CRFGradient extends Gradient {
  def compute(
    data: SparkVector,
    label: Double,
    weights: SparkVector,
    cumGradient: SparkVector
  ): Double = {
    throw new Exception("The original compute() method is not supported")
  }

  def computeCRF(sentences: Tagger, weights: BDV[Double], gradient: BDV[Double]): Double = {
    sentences.gradient(gradient, weights)
  }
}

trait UpdaterCRF extends Updater {
  def compute(
    weightsOld: SparkVector,
    gradient: SparkVector,
    stepSize: Double,
    iter: Int,
    regParam: Double
  ) = {
    throw new Exception("The original compute() method is not supported")
  }
  def computeCRF(weightsOld: BDV[Double], gradient: BDV[Double], regParam: Double): (BDV[Double], Double)
}

class L2Updater extends UpdaterCRF {
  def computeCRF(
    weightsOld: BDV[Double],
    gradient: BDV[Double],
    regParam: Double
  ): (BDV[Double], Double) = {
    val loss = Bsum(weightsOld :* weightsOld :* regParam)
    gradient :+= weightsOld :* (regParam * 2.0)
    (gradient, loss)
  }
}

class L1Updater extends UpdaterCRF {
  def computeCRF(
    weightsOld: BDV[Double],
    gradient: BDV[Double],
    regParam: Double
  ): (BDV[Double], Double) = {
    (gradient, 0.0)
  }
}

private class CostFun(
  taggers: RDD[Tagger],
  gradient: CRFGradient,
  updater: Updater,
  regParam: Double
) extends DiffFunction[BDV[Double]] with Serializable {

  override def calculate(weigths: BDV[Double]): (Double, BDV[Double]) = {

    val bcWeights = taggers.context.broadcast(weigths)
    val n = weigths.length
    lazy val treeDepth = math.max(math.ceil(math.log(taggers.partitions.length) / (math.log(2) * 2)).toInt, 2)

    val (expected, obj) = taggers.treeAggregate((BDV.zeros[Double](n), 0.0))(
      seqOp = (c, v) => {
      // c: (grad, obj), v: (sentence)
      val l = gradient.computeCRF(v, bcWeights.value, c._1)
      (c._1, c._2 + l)
    },
      combOp = (c1, c2) => {
      (c1._1 + c2._1, c1._2 + c2._2)
    }, treeDepth
    )

    bcWeights.destroy()

    val (grad, loss) = updater.asInstanceOf[UpdaterCRF].computeCRF(weigths, expected, regParam)
    (obj + loss, grad)
  }
}

