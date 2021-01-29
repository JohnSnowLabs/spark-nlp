package com.johnsnowlabs.ml.crf

import com.johnsnowlabs.ml.crf.VectorMath._
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class GradientSpec extends FlatSpec {
  val dataset = TestDatasets.small
  val instance = dataset.instances.head._2
  val metadata = dataset.metadata
  val features = metadata.attrFeatures.size + metadata.transitions.size
  val weights = Vector(features, 0.1f)

  val fb = new FbCalculator(2, metadata)
  val bruteForce = new BruteForceCalculator(metadata, fb)
  fb.calculate(instance, weights, 1f)


  "SGD" should "correctly calculates data estimation" taggedAs FastTest in {
    val instance = dataset.instances.head._2
    val labels = dataset.instances.head._1

    val features = dataset.metadata.attrFeatures.size + dataset.metadata.transitions.size
    val weights = Vector(features, 0.1f)
    fb.addObservedExpectations(weights, instance, labels, 0.1f)

    assert(weights.toSeq == Seq(0.2f, 0.2f, 0.3f, 0.2f, 0.3f, 0.4f, 0.2f, 0.2f))
  }


  "SGD" should "correctly calculates model estimation" taggedAs FastTest in {

    // 1. Calculate Model Expectation by Test BruteForce Algo
    val attrExp = metadata.attrFeatures.map{f =>
      val featureValues = instance.items.map(word => word.apply(f.attrId))
      bruteForce.calcExpectation(instance, f, featureValues)
    }

    val transExp = metadata.transitions.map(t =>
      bruteForce.calcExpectation(instance, t)
    )

    val a = -0.1f
    val expectations = (attrExp ++ transExp).toList
    val newWeights = expectations.map(e => 0.1f + a * e)

    // 2. Calculate Model Expectation by CRF Algo
    val weights = this.weights.clone()
    fb.addModelExpectations(weights, instance, a)

    // 3. Weights must be equal
    FloatAssert.seqEquals(weights.toSeq, newWeights)
  }

}
