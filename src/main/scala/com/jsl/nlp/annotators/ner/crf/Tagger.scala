package com.jsl.nlp.annotators.ner.crf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseVector => BDV, Vector => BV}

sealed trait Mode

private[nlp] case object LearnMode extends Mode

private[nlp] case object TestMode extends Mode

private[nlp] case class QueueElement(node: Node, fx: Double, gx: Double, next: QueueElement)

private[nlp] class Tagger(
  ySize: Int,
  mode: Mode
) extends Serializable {
  var nBest = 0
  var cost = 0.0
  var Z = 0.0
  var obj = 0.0
  var costFactor = 1.0
  val x = new ArrayBuffer[Array[String]]()
  val nodes = new ArrayBuffer[Node]()
  val answer = new ArrayBuffer[Int]()
  val result = new ArrayBuffer[Int]()
  val featureCache = new ArrayBuffer[Int]()
  val featureCacheIndex = new ArrayBuffer[Int]()
  val probMatrix = new ArrayBuffer[Double]()
  var seqProb = 0.0
  lazy val topN = ArrayBuffer.empty[Array[Int]]
  lazy val topNResult = ArrayBuffer.empty[Int]
  lazy val probN = ArrayBuffer.empty[Double]
  lazy val agenda = mutable.PriorityQueue.empty[QueueElement](
    Ordering.by((_: QueueElement).fx).reverse
  )

  def setCostFactor(costFactor: Double) = {
    this.costFactor = costFactor
    this
  }

  def setNBest(nBest: Int) = {
    this.nBest = nBest
    this
  }

  def read(lines: Sequence, feature_idx: FeatureIndex) = {
    var i = 0
    val tmpLines = lines.toArray
    while (i < tmpLines.length) {
      val t = tmpLines(i)
      mode match {
        case LearnMode =>
          var j = 0
          while (j < feature_idx.labels.length) {
            if (feature_idx.labels(j).equals(t.label)) {
              answer.append(j)
            }
            j += 1
          }
          x.append(t.tags)
        case TestMode =>
          x.append(t.tags)
          answer.append(0)
      }
      result.append(0)
      i += 1
    }
    this
  }

  /**
   * Set node relationship and its feature index.
   * Node represents a token.
   */
  def rebuildFeatures(): Unit = {

    nodes ++= Array.fill(x.length * ySize)(new Node)
    var i = 0
    while (i < nodes.length) {
      nodes(i).x = i / ySize
      nodes(i).y = i - nodes(i).x * ySize
      nodes(i).fVector = featureCacheIndex(nodes(i).x)
      i += 1
    }

    i = 0
    while (i < nodes.length) {
      if (nodes(i).x > 0) {
        val paths = Array.fill(ySize)(new Path)
        val nx = nodes(i).x
        val ny = nodes(i).y
        var j = 0
        while (j < paths.length) {
          paths(j).fVector = featureCacheIndex(nx + x.length - 1)
          paths(j).add((nx - 1) * ySize + ny, nx * ySize + j, nodes)
          j += 1
        }
      }
      i += 1
    }
  }

  /**
   * Calculate the expectation of each node
   */
  def forwardBackward(): Unit = {
    var i = 0
    while (i < nodes.length) {
      nodes(i).calcAlpha(nodes)
      i += 1
    }
    i = nodes.length - 1
    while (i >= 0) {
      nodes(i).calcBeta(nodes)
      i -= 1
    }
    Z = 0.0
    i = 0
    while (i < nodes.length) {
      if (nodes(i).x == 0) {
        Z = nodes(i).logSumExp(Z, nodes(i).beta, nodes(i).y == 0)
      }
      i += 1
    }
  }

  /**
   * Get the max expectation in the nodes and predicts the most likely label
   */
  def viterbi(): Unit = {
    var bestCost = Double.MinValue
    var best: Option[Node] = None

    var i = 0
    while (i < nodes.length) {
      bestCost = Double.MinValue
      best = None
      var j = 0
      while (j < nodes(i).lPath.length) {
        val p = nodes(i).lPath(j)
        val cost = nodes(p.lNode).bestCost + p.cost + nodes(i).cost
        if (cost > bestCost) {
          bestCost = cost
          best = Some(nodes(p.lNode))
        }
        j += 1
      }
      nodes(i).prev = best
      best match {
        case None =>
          nodes(i).bestCost = nodes(i).cost
        case _ =>
          nodes(i).bestCost = bestCost
      }
      i += 1
    }

    var nd: Option[Node] = Some(nodes.filter(_.x == x.length - 1).max(Ordering.by((_: Node).bestCost)))

    while (nd.isDefined) {
      result.update(nd.get.x, nd.get.y)
      nd = nd.get.prev
    }

    cost = -nodes((x.length - 1) * ySize + result.last).bestCost
  }

  def gradient(expected: BV[Double], alpha: BDV[Double]): Double = {

    buildLattice(alpha)
    forwardBackward()

    var i = 0
    while (i < nodes.length) {
      nodes(i).calExpectation(expected, Z, ySize, featureCache, nodes)
      i += 1
    }

    var s: Double = 0.0
    i = 0
    while (i < x.length) {
      var rIdx = nodes(i * ySize + answer(i)).fVector
      while (featureCache(rIdx) != -1) {
        expected(featureCache(rIdx) + answer(i)) -= 1.0
        rIdx += 1
      }
      s += nodes(i * ySize + answer(i)).cost
      var j = 0
      while (j < nodes(i * ySize + answer(i)).lPath.length) {
        val lNode = nodes(nodes(i * ySize + answer(i)).lPath(j).lNode)
        val rNode = nodes(nodes(i * ySize + answer(i)).lPath(j).rNode)
        val lPath = nodes(i * ySize + answer(i)).lPath(j)
        if (lNode.y == answer(lNode.x)) {
          rIdx = lPath.fVector
          while (featureCache(rIdx) != -1) {
            expected(featureCache(rIdx) + lNode.y * ySize + rNode.y) -= 1.0
            rIdx += 1
          }
          s += lPath.cost
        }
        j += 1
      }
      i += 1
    }

    viterbi()
    clear()
    Z - s
  }

  def probCalculate(): Unit = {
    probMatrix ++= Array.fill(x.length * ySize)(0.0)
    var idx: Int = 0
    var i = 0
    while (i < nodes.length) {
      val n = nodes(i)
      idx = n.x * ySize + n.y
      probMatrix(idx) = Math.exp(n.alpha + n.beta - n.cost - Z)
      i += 1
    }
    this.seqProb = Math.exp(-cost - Z)

  }

  def clear(): Unit = {
    var i = 0
    while (i < nodes.length) {
      nodes(i).lPath.clear()
      nodes(i).rPath.clear()
      i += 1
    }
    nodes.clear()
  }

  def parse(alpha: BDV[Double], mode: Option[VerboseMode]): Unit = {
    buildLattice(alpha)
    if (nBest > 0 || mode.isDefined) {
      forwardBackward()
      viterbi()
      probCalculate()
    } else
      viterbi()
    if (nBest > 0) {
      //initialize nBest
      if (agenda.nonEmpty) agenda.clear()
      val nodesTemp = nodes.slice((x.size - 1) * ySize, x.size * ySize)
      var i = 0
      while (i < nodesTemp.length) {
        val n = nodesTemp(i)
        agenda += QueueElement(n, -n.bestCost, -n.cost, null)
        i += 1
      }
      //find nBest
      i = 0
      while (i < this.nBest) {
        topNResult.clear()
        if (!nextNode)
          return
        probN.append(Math.exp(-cost - Z))
        topN.append(topNResult.toArray)
        i += 1
      }
    }
  }

  def buildLattice(alpha: BDV[Double]): Unit = {

    rebuildFeatures()
    var i = 0
    while (i < nodes.length) {
      calcCost(nodes(i), alpha)
      var j = 0
      while (j < nodes(i).lPath.length) {
        calcCost(nodes(i).lPath(j), alpha)
        j += 1
      }
      i += 1
    }

  }

  def calcCost(n: Node, alpha: BDV[Double]) = {
    var cd: Double = 0.0
    var idx: Int = n.fVector

    while (featureCache(idx) != -1) {
      cd += alpha(featureCache(idx) + n.y)
      idx += 1
    }
    n.cost = cd * costFactor
  }

  def calcCost(p: Path, alpha: BDV[Double]) = {
    var cd: Double = 0.0
    var idx: Int = p.fVector

    while (featureCache(idx) != -1) {
      cd += alpha(featureCache(idx) +
        nodes(p.lNode).y * ySize + nodes(p.rNode).y)
      idx += 1
    }
    p.cost = cd * costFactor
  }

  def nextNode: Boolean = {
    var top: QueueElement = null
    var rNode: Node = null
    while (agenda.nonEmpty) {
      top = agenda.dequeue()
      rNode = top.node
      if (rNode.x == 0) {
        var n: QueueElement = top
        var i = 0
        while (i < x.length) {
          topNResult.append(n.node.y)
          n = n.next
          i += 1
        }
        cost = top.gx
        return true
      }
      var i = 0
      while (i < rNode.lPath.length) {
        val p = rNode.lPath(i)
        val gx = -nodes(p.lNode).cost - p.cost + top.gx
        val fx = -nodes(p.lNode).bestCost - p.cost + top.gx
        agenda += QueueElement(nodes(p.lNode), fx, gx, top)
        i += 1
      }
    }
    false
  }
}
