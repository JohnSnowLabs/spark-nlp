package com.jsl.nlp.annotators

import com.jsl.nlp.{AnnotatorBuilder, SparkBasedTest, Document, Annotation}
import com.jsl.nlp.util.RegexRule
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait RegexMatcherBehaviors extends SparkBasedTest { this: FlatSpec =>

  def PredefinedRulesRegexMatcher(dataset: => Dataset[Row], rules: Seq[RegexRule]): Unit = {
    "A RegexMatcher Annotator" should "successfuly match submitted rules in text" in {
      println(dataset.schema)
      AnnotatorBuilder.withRegexMatcher(dataset, rules)
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
