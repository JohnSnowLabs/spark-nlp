package com.jsl.nlp.annotators

import com.jsl.nlp.util.regex.RegexRule
import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait RegexMatcherBehaviors { this: FlatSpec =>

  def predefinedRulesRegexMatcher(dataset: => Dataset[Row], rules: Seq[RegexRule]): Unit = {
    "A RegexMatcher Annotator" should s"successfuly match ${rules.map(_.value).mkString(",")}" in {
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
