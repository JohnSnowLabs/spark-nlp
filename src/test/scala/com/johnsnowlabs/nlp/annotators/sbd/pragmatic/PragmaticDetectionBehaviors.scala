package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

import scala.language.reflectiveCalls

trait PragmaticDetectionBehaviors { this: FlatSpec =>

  def fixture(dataset: => Dataset[Row]) = new {
    val df = AnnotatorBuilder.withFullPragmaticSentenceDetector(dataset)
    val documents = df.select("document")
    val sentences = df.select("sentence")
    val sentencesAnnotations = sentences
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { a => Annotation(a.getString(0), a.getInt(1), a.getInt(2), a.getString(3), a.getMap[String, String](4)) }
    val corpus = sentencesAnnotations
      .flatMap { a => a.result }
      .mkString("")
  }

  private def f1Score(result: Array[String], expected: Array[String]): Double = {
    val nMatches = result.count(expected.contains(_))
    val nOutput = result.length
    val nExpected = expected.length
    val precision = nMatches / nOutput.toDouble
    val recall = nMatches / nExpected.toDouble
    (2 * precision * recall) / (precision + recall)
  }

  def isolatedPDReadAndMatchResult(input: String, correctAnswer: Array[String], customBounds: Array[String] = Array.empty[String]): Unit = {
    s"pragmatic boundaries detector with ${input.take(10)}...:" should
      s"successfully identify sentences as ${correctAnswer.take(1).take(10).mkString}..." in {
      val pragmaticApproach = new MixedPragmaticMethod(true, customBounds)
      val result = pragmaticApproach.extractBounds(input)
      val diffInResult = result.map(_.content).diff(correctAnswer)
      val diffInCorrect = correctAnswer.diff(result.map(_.content))
      assert(
        result.map(_.content).sameElements(correctAnswer),
        s"\n--------------\nSENTENCE IS WRONG:\n--------------\n$input" +
        s"\n--------------\nBECAUSE RESULT:\n--------------\n@@${diffInResult.mkString("\n@@")}" +
          s"\n--------------\nIS NOT EXPECTED:\n--------------\n@@${diffInCorrect.mkString("\n@@")}")
      assert(result.forall(sentence => {
        sentence.end == sentence.start + sentence.content.length - 1
      }), "because length mismatch")
    }
  }

  def isolatedPDReadAndMatchResultTag(input: String, correctAnswer: Array[String], customBounds: Array[String] = Array.empty[String]): Unit = {
    s"pragmatic boundaries detector with ${input.take(10)}...:" should
      s"successfully identify sentences as ${correctAnswer.take(1).take(10).mkString}..." in {
      val result = new SentenceDetector().tag(input).map(_.content)
      val diffInResult = result.diff(correctAnswer)
      val diffInCorrect = correctAnswer.diff(result)
      assert(
        result.sameElements(correctAnswer),
        s"\n--------------\nSENTENCE IS WRONG:\n--------------\n$input" +
          s"\n--------------\nBECAUSE RESULT:\n--------------\n@@${diffInResult.mkString("\n@@")}" +
          s"\n--------------\nIS NOT EXPECTED:\n--------------\n@@${diffInCorrect.mkString("\n@@")}")
    }
  }

  def isolatedPDReadScore(input: String, correctAnswer: Array[String], customBounds: Array[String] = Array.empty[String]): Unit = {
    s"boundaries prediction" should s"have an F1 score higher than 95%" in {
      val pragmaticApproach = new MixedPragmaticMethod(true, customBounds)
      val result = pragmaticApproach.extractBounds(input).map(_.content)
      val f1 = f1Score(result, correctAnswer)
      val unmatched = result.zip(correctAnswer).toMap.mapValues("\n"+_)
      info(s"F1 Score is: $f1")
      assert(f1 > 0.95, s"F1 Score is below 95%.\nMatch sentences:\n${unmatched.mkString("\n")}")
    }
  }

  def sparkBasedSentenceDetector(dataset: => Dataset[Row]): Unit = {
    "a Pragmatic Sentence Detection Annotator" should s"successfully annotate documents" in {
      val f = fixture(dataset)
      assert(f.sentencesAnnotations.nonEmpty, "Annotations should exists")
    }

    it should "add annotators of type sbd" in {
      val f = fixture(dataset)
      f.sentencesAnnotations.foreach { a =>
        assert(a.annotatorType == AnnotatorType.DOCUMENT, "annotatorType should sbd")
      }
    }
  }
}
