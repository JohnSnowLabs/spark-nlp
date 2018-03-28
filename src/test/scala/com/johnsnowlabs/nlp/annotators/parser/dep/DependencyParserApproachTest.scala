package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp._
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec
import scala.language.reflectiveCalls

class DependencyParserApproachTest extends FlatSpec {
  def fixture = new {
    val df = AnnotatorBuilder.withDependencyParser(DataBuilder.basicDataBuild(ContentProvider.depSentence))
    val dependencies = df.select("dependency")
    val depAnnotations = dependencies
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
    val tokens = df.select("token")
    val tokenAnnotations = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
  }

  "A DependencyParser" should "add annotations" in {
    val f = fixture
    assert(f.dependencies.count > 0, "Annotations count should be greater than 0")
  }

  it should "add annotations with the correct annotationType" in {
    val f = fixture
    f.depAnnotations.foreach { a =>
       assert(a.annotatorType == AnnotatorType.DEPENDENCY, s"Annotation type should ${AnnotatorType.DEPENDENCY}")
    }
  }

  it should "annotate each token" in {
    val f = fixture
    assert(f.tokenAnnotations.size == f.depAnnotations.size, s"Every token should be annotated")
  }

  it should "annotate each word with a head" in {
    val f = fixture
    f.depAnnotations.foreach { a =>
      assert(a.result.nonEmpty, s"Result should have a head")
    }
  }

  it should "annotate each word with the correct indexes" in {
    val f = fixture
    f.depAnnotations
      .zip(f.tokenAnnotations)
      .foreach { case (dep, token) => assert(dep.begin == token.begin && dep.end == token.end, s"Token and word should have equal indixes") }
  }
}
