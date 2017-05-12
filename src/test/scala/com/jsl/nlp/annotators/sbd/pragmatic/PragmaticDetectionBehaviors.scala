package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.SparkBasedTest
import org.scalatest.FlatSpec

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait PragmaticDetectionBehaviors extends SparkBasedTest { this: FlatSpec =>

  private def f1Score(result: Array[String], expected: Array[String]): Double = {
    val nMatches = result.count(expected.contains(_))
    val nOutput = result.length
    val nExpected = expected.length
    val precision = nMatches / nOutput.toDouble
    val recall = nMatches / nExpected.toDouble
    (2 * precision * recall) / (precision + recall)
  }

  def isolatedPDReadAndMatchResult(input: String, correctAnswer: Array[String], computeF1Score: Boolean = false): Unit = {
    s"text: $input" should s"successfully identify the following sentences:${correctAnswer.mkString("@@")}" in {
      val result = new PragmaticDetection(input)
        .prepare
        .extract
        .map(_.content)
      if (computeF1Score) {
        val f1 = f1Score(result, correctAnswer)
        assert(f1 > 0.5, "F1 Score is below 50%")
      }
      assert(result.sameElements(correctAnswer), s"\nRESULT: ${result.mkString("@@")} IS NOT: ${correctAnswer.mkString("@@")}")
    }
  }

}
