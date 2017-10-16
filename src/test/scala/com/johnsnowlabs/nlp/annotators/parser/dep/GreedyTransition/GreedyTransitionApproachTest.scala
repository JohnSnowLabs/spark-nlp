package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, ContentProvider, DataBuilder}
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec
import scala.language.reflectiveCalls
import org.scalatest.Matchers._

class GreedyTransitionApproachTest extends FlatSpec {
  def fixture = new {
    val model = new GreedyTransitionApproach
    val df = AnnotatorBuilder.withFullPOSTagger(DataBuilder.basicDataBuild(ContentProvider.depSentence))
    val tokens = df.select("token")
    val tokenAnnotations = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getString(1), r.getMap[String, String](2))
      }
      .sortBy { _.metadata(Annotation.BEGIN).toInt }
    val posTags = df.select("pos")
    val posTagAnnotations = posTags
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getString(1), r.getMap[String, String](2))
      }
      .sortBy { _.metadata(Annotation.BEGIN).toInt }
  }

  "A GreedyTransitionApproach" should "return an array of dependencies" in {
    val f = fixture
    assert(f.model.parse(f.tokenAnnotations, f.posTagAnnotations).size == f.tokenAnnotations.size)
  }
}
