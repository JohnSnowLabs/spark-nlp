package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.SparkBasedTest
import org.scalatest.FlatSpec

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait PragmaticDetectionBehaviors extends SparkBasedTest { this: FlatSpec =>

  def isolatedPDReadAndMatchResult(input: String, correctAnswer: Array[String], computeF1Score: Boolean = false): Unit = {
    s"text: ${input}" should s"successfully identify the following sentences:${correctAnswer.mkString("@@")}" in {
      val result = new PragmaticDetection(input)
        .prepare
        .extract
        .map(_.content)
      assert(result.sameElements(correctAnswer), s"\nRESULT: ${result.mkString("@@")} IS NOT: ${correctAnswer.mkString("@@")}")
    }
  }

}
