package com.johnsnowlabs.nlp.annotators.ner.regex

import org.scalatest._
import org.apache.spark.sql.{Dataset, Row}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.nlp.AnnotatorType._
import scala.language.reflectiveCalls

trait RegexApproachBehaviors { this: FlatSpec =>
  def fixture(dataset: Dataset[Row]) = new {
    val df = AnnotatorBuilder.withNERTagger(dataset)
    val nerAnnotations = df.select("ner")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  def isolatedDictionaryTagging(
                                 trainedTagger: NERRegexModel,
                                 targetSentences: Array[String]
  ): Unit = {
    s"Dictionary tagger" should "successfully tag the entities" in {
      trainedTagger.tag(targetSentences)
    }
  }

  def sparkBasedNERTagger(dataset: => Dataset[Row]): Unit = {
    "a Dictionary NER tagger Annotator" should s"successfully tag words " in {
      val f = fixture(dataset)
      assert(f.nerAnnotations.size > 0, "Annotations should exist")
    }

    it should "tag words with the appropiate annotator type" in {
      val f = fixture(dataset)
      f.nerAnnotations.foreach { a => assert(a.annotatorType == NAMED_ENTITY) }
    }
  }
}
