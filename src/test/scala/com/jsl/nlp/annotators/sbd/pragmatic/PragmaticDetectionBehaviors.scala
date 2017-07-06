package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.{AnnotatorBuilder, Document, Annotation}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait PragmaticDetectionBehaviors { this: FlatSpec =>

  private def f1Score(result: Array[String], expected: Array[String]): Double = {
    val nMatches = result.count(expected.contains(_))
    val nOutput = result.length
    val nExpected = expected.length
    val precision = nMatches / nOutput.toDouble
    val recall = nMatches / nExpected.toDouble
    (2 * precision * recall) / (precision + recall)
  }

  def isolatedPDReadAndMatchResult(input: String, correctAnswer: Array[String]): Unit = {
    require(input == correctAnswer.mkString(" "),
      s"provided bad test input\ninput:\n$input\nand correct answer:\n${correctAnswer.mkString(" ")}\ndo not match?")
    s"pragmatic boundaries detector with ${input.take(10)}...:" should
      s"successfully identify sentences as ${correctAnswer.take(1).take(10).mkString}..." in {
      val pragmaticApproach = new PragmaticApproach
      val result = pragmaticApproach.extractBounds(input).map(_.content)
      val diffInResult = result.diff(correctAnswer)
      val diffInCorrect = correctAnswer.diff(result)
      assert(
        result.sameElements(correctAnswer),
        s"\n--------------\nBECAUSE RESULT:\n--------------\n@@${diffInResult.mkString("\n@@")}" +
          s"\n--------------\nIS NOT EXPECTED:\n--------------\n@@${diffInCorrect.mkString("\n@@")}")
    }
  }

  def isolatedPDReadScore(input: String, correctAnswer: Array[String]): Unit = {
    s"boundaries prediction" should s"have an F1 score higher than 95%" in {
      val pragmaticApproach= new PragmaticApproach
      val result = pragmaticApproach.extractBounds(input).map(_.content)
      val f1 = f1Score(result, correctAnswer)
      val unmatched = result.zip(correctAnswer).toMap.mapValues("\n"+_)
      info(s"F1 Score is: $f1")
      assert(f1 > 0.95, s"F1 Score is below 95%.\nMatch sentences:\n${unmatched.mkString("\n")}")
    }
  }

  def sparkBasedSentenceDetector(dataset: => Dataset[Row]): Unit = {
    "a Pragmatic Sentence Detection Annotator" should s"successfully split sentences " in {
      info(dataset.schema.mkString(","))
      AnnotatorBuilder.withFullPragmaticSentenceDetector(dataset)
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](1).map(Annotation(_)).foreach {
            matchedAnnotation =>
              println(matchedAnnotation, matchedAnnotation.metadata.mkString(","))
          }
      }
    }
  }

}
