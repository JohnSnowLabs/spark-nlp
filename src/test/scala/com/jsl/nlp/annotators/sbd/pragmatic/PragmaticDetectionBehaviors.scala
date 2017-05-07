package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.SparkBasedTest
import org.scalatest.FlatSpec

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait PragmaticDetectionBehaviors extends SparkBasedTest { this: FlatSpec =>

  def isolatedPDRead(input: => String, correctAnswer: Array[String]): Unit = {
    "an input string" should "successfully parse sentence boundaries" in {
      val result = new PragmaticDetection(input)
        .prepare
        .extract
      assert(result.map(_.content).sameElements(correctAnswer))
    }
  }

}
