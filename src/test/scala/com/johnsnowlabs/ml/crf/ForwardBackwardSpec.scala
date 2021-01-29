package com.johnsnowlabs.ml.crf

import com.johnsnowlabs.ml.crf.VectorMath._
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class ForwardBackwardSpec extends FlatSpec {

  val dataset = TestDatasets.small
  val instance = dataset.instances.head._2
  val metadata = dataset.metadata
  val labels = metadata.label2Id.size
  val features = metadata.attrFeatures.size + metadata.transitions.size
  val weights = Vector(features, 0.1f)

  val fb = new FbCalculator(2, metadata)
  val bruteForce = new BruteForceCalculator(metadata, fb)
  fb.calculate(instance, weights, 1f)

  "EdgeCalculator" should "fill matrix correctly for first word" taggedAs FastTest in {
    val firstWordFeatures = dataset.instances(0)._2.items(0).values
    val logEdge = Matrix(3, 3)

    EdgeCalculator.fillLogEdges(firstWordFeatures, weights, 1f, metadata, logEdge)

    assert(logEdge(0).toSeq == Seq(0f, 0.5f, 0.3f))
    assert(logEdge(1).toSeq == Seq(0f, 0.4f, 0.4f))
    assert(logEdge(2).toSeq == Seq(0f, 0.4f, 0.3f))
  }


  "EdgeCalculate" should "fill matrix correctly for second word" taggedAs FastTest in {
    val secondWordFeatures = dataset.instances(0)._2.items(1).values
    val logEdge = Matrix(3, 3)

    EdgeCalculator.fillLogEdges(secondWordFeatures, weights, 1f, metadata, logEdge)
    FloatAssert.seqEquals(logEdge(0), Seq(0f, 0.6f, 0.6f))
    FloatAssert.seqEquals(logEdge(1), Seq(0f, 0.5f, 0.7f))
    FloatAssert.seqEquals(logEdge(2), Seq(0f, 0.5f, 0.6f))
  }


  "EdgeCalculator" should "fill matrix correctly according to scale param" taggedAs FastTest in {
    val secondWordFeatures = dataset.instances(0)._2.items(1).values
    val logEdge = Matrix(3, 3)

    val weights = Vector(features, 1f)

    EdgeCalculator.fillLogEdges(secondWordFeatures, weights, 0.1f, metadata, logEdge)
    FloatAssert.seqEquals(logEdge(0), Seq(0f, 0.6f, 0.6f))
    FloatAssert.seqEquals(logEdge(1), Seq(0f, 0.5f, 0.7f))
    FloatAssert.seqEquals(logEdge(2), Seq(0f, 0.5f, 0.6f))
  }


  "FbCalculator" should "calculate c correct" taggedAs FastTest in {
    // Calculate Z(x) as expected
    val zTest = fb.c.reduce(_*_)

    // Calculate Z(x) by brute force
    val z = fb.phi
      .reduce((a, b) => mult(a, b))(0)
      .sum
    assert(zTest == z)

    // Calculate Z(x) from alpha
    val alphaPaths = fb.alpha(instance.items.length - 1).sum
    assert(alphaPaths == 1f)

    // Calculate Z(x) from beta
    val betaPaths = fb.beta(0).zip(fb.phi(0)(0)).map{case(a,b) => a*b}.sum
    assert(betaPaths == 1f)
  }


  "FbCalculator" should "calculated alpha and beta should satisfy invariants" taggedAs FastTest in {
    for (i <- 0 until instance.items.size) {
      val fbZ = Range(0, labels).map(label => fb.alpha(i)(label) * fb.beta(i)(label)*fb.c(i)).sum
      assert(fbZ == 1f)
    }
  }

  "FbCalculator" should "calculate phi and logPhi correctly" taggedAs FastTest in {
    for (i <- 0 until instance.items.size) {
      for (from <- 0 until labels) {
        for (to <- 0 until labels) {
          assert(fb.phi(i)(from)(to) == Math.exp(fb.logPhi(i)(from)(to)).toFloat)
        }
      }
    }
  }


  "FbCalculator" should "correctly estimates paths goes through label at time" taggedAs FastTest in {
    for (i <- 0 until instance.items.length) {
      for (label <- 0 until labels) {
        val fBProbe = fb.alpha(i)(label) * fb.beta(i)(label) * fb.c(i)
        val probe = bruteForce.getProbe(instance, i, label)
        FloatAssert.equals(fBProbe, probe)
      }
    }
  }
}
