package com.jsl.nlp.annotators.ner.crf

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{Vector => BV}

private[nlp] class Node extends Serializable {
  var x = 0
  var y = 0
  var alpha = 0.0
  var beta = 0.0
  var cost = 0.0
  var bestCost = 0.0
  var prev: Option[Node] = None
  var fVector = 0
  val lPath = new ArrayBuffer[Path]()
  val rPath = new ArrayBuffer[Path]()

  /**
   * simplify the log likelihood.
   */
  def logSumExp(x: Double, y: Double, flg: Boolean): Double = {
    val MINUS_LOG_EPSILON = 50.0
    if (flg) y
    else {
      val vMin: Double = math.min(x, y)
      val vMax: Double = math.max(x, y)
      if (vMax > vMin + MINUS_LOG_EPSILON) vMax else vMax + math.log(math.exp(vMin - vMax) + 1.0)
    }
  }

  def calcAlpha(nodes: ArrayBuffer[Node]): Unit = {
    alpha = 0.0
    var i = 0
    while (i < lPath.length) {
      alpha = logSumExp(alpha, lPath(i).cost + nodes(lPath(i).lNode).alpha, i == 0)
      i += 1
    }
    alpha += cost
  }

  def calcBeta(nodes: ArrayBuffer[Node]): Unit = {
    beta = 0.0
    var i = 0
    while (i < rPath.length) {
      beta = logSumExp(beta, rPath(i).cost + nodes(rPath(i).rNode).beta, i == 0)
      i += 1
    }
    beta += cost
  }

  def calExpectation(
    expected: BV[Double],
    Z: Double,
    size: Int,
    featureCache: ArrayBuffer[Int],
    nodes: ArrayBuffer[Node]
  ): Unit = {
    val c: Double = math.exp(alpha + beta - cost - Z)

    var idx: Int = fVector
    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + y) += c
      idx += 1
    }

    var i = 0
    while (i < lPath.length) {
      lPath(i).calExpectation(expected, Z, size, featureCache, nodes)
      i += 1
    }

  }
}

private[nlp] class Path extends Serializable {
  var rNode = 0
  var lNode = 0
  var cost = 0.0
  var fVector = 0

  def calExpectation(
    expected: BV[Double],
    Z: Double,
    size: Int,
    featureCache: ArrayBuffer[Int],
    nodes: ArrayBuffer[Node]
  ): Unit = {
    val c: Double = math.exp(nodes(lNode).alpha + cost + nodes(rNode).beta - Z)
    var idx: Int = fVector

    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + nodes(lNode).y * size + nodes(rNode).y) += c
      idx += 1
    }
  }

  def add(lnd: Int, rnd: Int, nodes: ArrayBuffer[Node]): Unit = {
    lNode = lnd
    rNode = rnd
    nodes(lNode).rPath.append(this)
    nodes(rNode).lPath.append(this)
  }
}
