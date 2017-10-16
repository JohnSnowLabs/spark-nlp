package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

import scala.language.reflectiveCalls

trait RegexTokenizerBehaviors { this: FlatSpec =>

  def fixture(dataset: => Dataset[Row]) = new {
    val df = AnnotatorBuilder.withTokenizer(AnnotatorBuilder.withFullPragmaticSentenceDetector(dataset))
    val documents = df.select("document")
    val sentences = df.select("sentence")
    val tokens = df.select("token")
    val sentencesAnnotations = sentences
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { a => Annotation(a.getString(0), a.getString(1), a.getMap[String, String](2)) }
    val tokensAnnotations = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0)}
      .map { a => Annotation(a.getString(0), a.getString(1), a.getMap[String, String](2)) }
    val corpus = sentencesAnnotations
      .map { a => a.result }
      .mkString("")
  }

  def fullTokenizerPipeline(dataset: => Dataset[Row]) {
    "A RegexTokenizer Annotator" should "successfully transform data" in {
      val f = fixture(dataset)
      assert(f.tokensAnnotations.nonEmpty, "RegexTokenizer should add annotators")
    }

    it should "annotate using the annotatorType of token" in {
      val f = fixture(dataset)
      assert(f.tokensAnnotations.nonEmpty, "RegexTokenizer should add annotators")
      f.tokensAnnotations.foreach { a =>
        assert(a.annotatorType == AnnotatorType.TOKEN, "RegexTokenizer annotations type should be equal to 'token'")
      }
    }

    it should "annotate with the correct word indexes" in {
      val f = fixture(dataset)
      f.tokensAnnotations.foreach { a =>
        val token = a.result
        val sentenceToken = f.corpus.slice(a.metadata(Annotation.BEGIN).toInt, a.metadata(Annotation.END).toInt + 1)
        assert(sentenceToken == token, s"Word ($sentenceToken) from sentence at (${a.metadata(Annotation.BEGIN).toInt},${a.metadata(Annotation.END).toInt}) should be equal to token ($token) inside the corpus ${f.corpus}")
      }
    }
  }
}
