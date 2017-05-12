package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.SparkBasedTest
import org.scalatest._

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

  def isolatedPDReadAndMatchResult(input: String, correctAnswer: Array[String]): Unit = {
    s"text: $input" should s"successfully identify the following sentences:${correctAnswer.mkString("@@")}" in {
      val result = new PragmaticDetection(input)
        .prepare
        .extract
        .map(_.content)
      assert(result.sameElements(correctAnswer), s"\nRESULT: ${result.mkString("@@")} IS NOT: ${correctAnswer.mkString("@@")}")
    }
  }

  def isolatedPDReadScore(input: String, correctAnswer: Array[String]): Unit = {
    s"text: $input" should s"have an F1 score higher than 50%" in {
      val result = new PragmaticDetection(input)
        .prepare
        .extract
        .map(_.content)
      val f1 = f1Score(result, correctAnswer)
      val unmatched = result.zip(correctAnswer).toMap.mapValues("\n"+_)
      info(s"F1 Score is: $f1")
      assert(f1 > 0.5, s"F1 Score is below 50%.\nMatch sentences:\n${unmatched.mkString("\n")}")
    }
  }

}
